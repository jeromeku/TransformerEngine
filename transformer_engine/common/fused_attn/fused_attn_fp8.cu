/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <atomic>

#include "../common.h"
#include "../cudnn_utils.h"
#include "../util/system.h"
#include "fused_attn_fp8.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

namespace {

// Round a THD sequence extent up to a bucket, so the graph cache key is stable across micro-batches
// whose longest document differs. Guarantees s <= bucket(s) <= 2 * s, so the declared extent stays
// within a factor of two of the real one. The bound is tight only at the floor, where bucket(1) is
// 2; for every s > 1 the upper inequality is strict. The bound matters because the FP8 workspace is
// sized from the declared extent, so every token declared beyond the real ones is paid for in
// memory. No cost model is assumed here beyond that direction; the workspace tests bound it by
// comparing measurements rather than by predicting them.
//
// get_max_tokens is not usable here even though the buckets at and above 1024 are its own. It floors
// every input from 1 through 1024 to 1024, which is right for a total token count and wrong for a
// per-sequence extent: it describes a batch of one-token documents as 1024 tokens each, and the
// resulting allocation can exceed device memory outright. Below 1024 this rounds to a power of two
// instead, so a short document stays short.
//
// The floor of 2 is measured, not chosen: cuDNN reports "No valid execution plans" for a declared
// extent of 1, and builds normally at 2. Checked at three geometries -- 32 heads with 8 GQA groups,
// 16-head MHA, and 64 heads with 8 GQA groups, all head dim 128 with a padding-causal mask -- and
// the floor did not move. Other geometries are not covered by that measurement; if one of them
// needs a larger minimum it will surface as the same cuDNN error rather than as a wrong answer.
//
// Buckets at and above 1024 are unchanged from get_max_tokens, so packed rows of production size
// land on exactly the buckets they did before.
size_t fp8_thd_sequence_bucket(int64_t sequence_extent) {
  NVTE_CHECK(sequence_extent > 0, "FP8 THD sequence extent must be positive, got ", sequence_extent,
             ".");
  constexpr size_t min_bucket = 2;
  if (sequence_extent >= 1024) {
    return get_max_tokens(static_cast<size_t>(sequence_extent));
  }
  size_t bucket = min_bucket;
  while (bucket < static_cast<size_t>(sequence_extent)) {
    bucket <<= 1;
  }
  return bucket;
}

}  // namespace

// Counters for the two FP8 graph caches, for tests and diagnosis.
//
// The caches themselves are function-local `static thread_local`, so nothing outside can see their
// size; these mirror it.
//
// Off unless NVTE_FP8_ATTN_CACHE_STATS is set. This is diagnostic instrumentation only -- nothing
// in the implementation reads a counter, and no control flow depends on one -- so it should not be
// in the production path at all. When disabled the cost is a load and a not-taken branch against a
// cached bool; when enabled, a relaxed atomic increment measured at 6.2 ns against an attention
// call of several milliseconds.
//
// Process-wide and atomic, not thread-local, even though the caches are. PyTorch runs backward on
// an autograd worker thread, so thread-local counters are written there and read as zero from the
// thread that ran the forward -- reporting "no backward graphs were ever built", which is exactly
// the false reassurance this exists to prevent.
//
// Consequence of the caches staying thread-local: `entries` is the size of the last cache written,
// not a sum over threads. With one forward thread and one autograd thread -- the normal case --
// fprop_entries and bprop_entries each describe their own cache and are exact.
bool fp8_cache_stats_enabled() {
  static const bool enabled = transformer_engine::getenv<bool>("NVTE_FP8_ATTN_CACHE_STATS", false);
  return enabled;
}

struct FP8CacheCounters {
  std::atomic<size_t> fprop_lookups{0};
  std::atomic<size_t> fprop_hits{0};
  std::atomic<size_t> fprop_entries{0};
  std::atomic<size_t> bprop_lookups{0};
  std::atomic<size_t> bprop_hits{0};
  std::atomic<size_t> bprop_entries{0};
};
FP8CacheCounters fp8_cache_stats;

// fused attention FWD FP8 with FE 1.0+
void fused_attn_fp8_fwd_impl(
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d_qk, int64_t d_v,
    bool is_training, float scaling_factor, float dropout_probability, NVTE_QKV_Layout qkv_layout,
    NVTE_QKV_Format o_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    NVTE_Softmax_Type softmax_type, int64_t window_size_left, int64_t window_size_right,
    bool bottom_right_diagonal, void* devPtrQ, void* devPtrK, void* devPtrV,
    void* devPtrSoftmaxOffset, void* devPtrM, void* devPtrO, void* devPtrDescaleQ,
    void* devPtrDescaleK, void* devPtrDescaleV, void* devPtrDescaleS, void* devPtrScaleS,
    void* devPtrScaleO, void* devPtrAmaxO, void* devPtrAmaxS, void* devPtrcuSeqlensQ,
    void* devPtrcuSeqlensKV, void* devPtrcuSeqlensQPadded, void* devPtrcuSeqlensKVPadded,
    void* devPtrDropoutSeed, void* devPtrDropoutOffset, cudnn_frontend::DataType_t qkv_tensor_type,
    cudnn_frontend::DataType_t o_tensor_type, NVTEScalingMode scaling_mode,
    NVTE_QKV_Format qkv_scale_inv_format, void* workspace, size_t* workspace_size,
    cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  const auto cudnn_runtime_version = cudnnGetVersion();
  bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
  bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
  bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
                    (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_dropout = (is_training && dropout_probability != 0.0f);
  bool is_softmax_offset = (softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX);
  auto bias_b = b;
  auto bias_h = h;
  auto bias_sq = s_q;
  auto bias_skv = s_kv;
  NVTE_CHECK(~is_bias, "FP8 fused attention does not support pre/post_scale_bias yet!");
  NVTE_CHECK(~is_alibi, "FP8 fused attention does not support ALiBi yet!");
  bool is_delayed_scaling = (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) &&
                            (o_tensor_type == cudnn_frontend::DataType_t::FP8_E4M3 ||
                             o_tensor_type == cudnn_frontend::DataType_t::FP8_E5M2);
  bool is_current_scaling = (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) &&
                            (o_tensor_type == cudnn_frontend::DataType_t::HALF ||
                             o_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);
  bool is_mxfp8 = (scaling_mode == NVTE_MXFP8_1D_SCALING) &&
                  (o_tensor_type == cudnn_frontend::DataType_t::HALF ||
                   o_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);
  NVTE_CHECK(
      is_delayed_scaling || is_current_scaling || is_mxfp8,
      "FP8 fused attention only supports FP8DelayedScaling or FP8CurrentScaling or MXFP8 recipes!");
  NVTE_CHECK(!is_mxfp8 || cudnn_runtime_version >= 92100,
             "MXFP8 fused attention requires cuDNN 9.21.0 or later!");

  // THD (packed varlen): element (ragged) base offsets are built from cu_seqlens_*_padded (physical
  // slot boundaries), while per-document extents for masking come from cu_seqlens (actual lengths).
  // The two coincide for a gap-free row, so contiguous packing is unchanged; a physical inter-
  // sequence gap addresses each document at its padded base rather than reading a neighbour's tokens.
  const bool is_ragged = (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD);
  NVTE_CHECK(!is_ragged || is_padding,
             "FP8 fused attention with THD requires a padding or padding_causal mask!");

  // cu_seqlens carries [actual_b + 1] valid entries. b is not quantized (see below), so this is
  // currently equal to b; the two are kept distinct because the conversion kernels take both and
  // passing the real one is the correct call regardless.
  //
  // This is not by itself sufficient to introduce a batch capacity later. cu_seqlens_padded_to_
  // offsets derives V offsets for the interleaved layouts by reading offsets_k[cu_seqlens_id]
  // within the same kernel, so with tid > actual_b several threads would read the terminal
  // offsets_k entry while the actual_b thread writes it. Unreachable while actual_b == b. Before
  // raising b: add a conversion-kernel test with actual_b < max_b for every packed layout group,
  // and remove or order that same-kernel read dependency.
  const int64_t actual_b = b;
  if (is_ragged) {
    // Round the sequence extents up to buckets so the graph cache key is stable. Under THD, s_q and
    // s_kv arrive as the longest document in the micro-batch, which changes nearly every call, and
    // FADescriptor_v1 orders the cache on them, so without this a fresh cuDNN graph is built almost
    // every call. Rounding up changes no result because both are declared upper bounds; the true
    // per-document extents reach the kernel through cu_seqlens and the ragged offsets.
    //
    // Only the sequence extents, not the batch. The F16 ragged path also replaces b with a bucketed
    // capacity, which costs a flash-style kernel nothing, but the FP8 workspace is sized from the
    // declared batch as well as the declared extent -- so declaring a capacity rather than the true
    // document count multiplies it. Leaving b exact keeps the workspace proportional to the tokens
    // actually present.
    NVTE_CHECK(actual_b > 0, "FP8 THD attention requires a positive batch size.");
    const int64_t bucketed_s_q = static_cast<int64_t>(fp8_thd_sequence_bucket(s_q));
    const int64_t bucketed_s_kv = static_cast<int64_t>(fp8_thd_sequence_bucket(s_kv));
    NVTE_CHECK(bucketed_s_q >= s_q, "FP8 THD query bucket ", bucketed_s_q,
               " is below the longest query sequence ", s_q, ".");
    NVTE_CHECK(bucketed_s_kv >= s_kv, "FP8 THD key/value bucket ", bucketed_s_kv,
               " is below the longest key/value sequence ", s_kv, ".");
    s_q = bucketed_s_q;
    s_kv = bucketed_s_kv;
    bias_sq = s_q;
    bias_skv = s_kv;
  }

  // Match the F16 path: 64-bit wherever the runtime allows, rather than deriving the width from
  // problem size.
  const DType ragged_offset_type = cudnn_runtime_version >= 90500 ? DType::kInt64 : DType::kInt32;

  try {
    FADescriptor_v1 descriptor{b,
                               h,
                               hg,
                               s_q,
                               s_kv,
                               d_qk,
                               d_v,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               bias_b,
                               bias_h,
                               bias_sq,
                               bias_skv,
                               scaling_factor,
                               is_training,
                               dropout_probability,
                               qkv_layout,
                               o_format,
                               NVTE_QKV_Format_NOT_SET,
                               NVTE_QKV_Layout_NOT_SET,
                               qkv_scale_inv_format,
                               NVTE_QKV_Format_NOT_SET,
                               bias_type,
                               mask_type,
                               softmax_type,
                               window_size_left,
                               window_size_right,
                               bottom_right_diagonal,
                               true,
                               qkv_tensor_type,
                               o_tensor_type,
                               cudnn_frontend::DataType_t::NOT_SET,
                               cudnn_frontend::DataType_t::NOT_SET,
                               false};

    namespace fe = cudnn_frontend;
    using graph_and_tensors =
        std::tuple<std::shared_ptr<fe::graph::Graph>,
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // K
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // V
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // attn_scale
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // O
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // bias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // softmax_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_o
                   std::shared_ptr<fe::graph::Tensor_attributes>>;  // offset_stats

    using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
    static thread_local CacheType sdpa_fp8_fprop_cache;

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph = [&](CacheType& cache, const FADescriptor_v1& descriptor) -> graph_and_tensors {
      // if hit, return
      if (fp8_cache_stats_enabled())
        fp8_cache_stats.fprop_lookups.fetch_add(1, std::memory_order_relaxed);
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        if (fp8_cache_stats_enabled())
          fp8_cache_stats.fprop_hits.fetch_add(1, std::memory_order_relaxed);
        auto graph = it->second;
        return graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();
      mha_graph->set_io_data_type(qkv_tensor_type)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, attn_scale;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_q, descale_k, descale_v;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_s, scale_s, scale_o;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, softmax_offset, seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;
      std::shared_ptr<fe::graph::Tensor_attributes> offset_q, offset_k, offset_v, offset_o,
          offset_stats;

      // Q, K, V, attn_scale
      std::vector<int64_t> q_strides(4), k_strides(4), v_strides(4);
      generateMatrixStridesWithLayout(b, h, hg, s_q, s_kv, d_qk, d_v, q_strides.data(),
                                      k_strides.data(), v_strides.data(), qkv_layout);
      Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q")
                                .set_dim({b, h, s_q, d_qk})
                                .set_stride(q_strides)
                                .set_data_type(qkv_tensor_type));
      K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K")
                                .set_dim({b, hg, s_kv, d_qk})
                                .set_stride(k_strides)
                                .set_data_type(qkv_tensor_type));
      V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V")
                                .set_dim({b, hg, s_kv, d_v})
                                .set_stride(v_strides)
                                .set_data_type(qkv_tensor_type));
      // THD (packed varlen): Q/K/V/O/Stats keep their dense {b,h,s,d} dims and BSHD-style
      // strides, and a per-batch ragged offset displaces each sequence's base pointer into the
      // packed buffer. cuDNN requires a padding-family mask alongside these
      // (cudnn_frontend sdpa_support_surface.h), which the selector already enforces.
      if (is_ragged) {
        auto make_offset = [&](const char* name) {
          return mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name(name)
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        };
        offset_q = make_offset("offset_q");
        offset_k = make_offset("offset_k");
        offset_v = make_offset("offset_v");
        offset_o = make_offset("offset_o");
        offset_stats = make_offset("offset_stats");
        Q->set_ragged_offset(offset_q);
        K->set_ragged_offset(offset_k);
        V->set_ragged_offset(offset_v);
      }

      attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("attn_scale")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_is_pass_by_value(true)
                                         .set_data_type(fe::DataType_t::FLOAT));

      // Descale_q, Descale_k, Descale_v, Descale_s, Scale_s, Scale_o
      if (is_delayed_scaling || is_current_scaling) {
        descale_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("Descale_q")
                                          .set_dim({1, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::FLOAT));
        descale_k = mha_graph->tensor_like(descale_q, "Descale_q");
        descale_v = mha_graph->tensor_like(descale_q, "Descale_v");
        descale_s = mha_graph->tensor_like(descale_q, "Descale_s");
        scale_s = mha_graph->tensor_like(descale_q, "Scale_s");
        if (is_delayed_scaling) {
          scale_o = mha_graph->tensor_like(descale_q, "Scale_o");
        }
        if (is_current_scaling) {
          scale_o = mha_graph->tensor(1.0f);
        }
      } else if (is_mxfp8) {
        NVTE_QKV_Format q_scale_inv_format = (qkv_scale_inv_format != NVTE_QKV_Format_NOT_SET)
                                                 ? qkv_scale_inv_format
                                                 : nvte_get_q_format(qkv_layout);
        NVTE_QKV_Format kv_scale_inv_format = (qkv_scale_inv_format != NVTE_QKV_Format_NOT_SET)
                                                  ? qkv_scale_inv_format
                                                  : nvte_get_kv_format(qkv_layout);
        std::vector<int64_t> q_scale_strides(4);
        std::vector<int64_t> k_scale_strides(4);
        std::vector<int64_t> v_scale_strides(4);
        auto padded = pad_s_d_for_mxfp8(s_q, s_kv, d_qk, d_v);
        generateMatrixStridesWithFormat(b, h, padded.s_q_padded, padded.d_qk_scale_padded,
                                        q_scale_strides.data(), q_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_padded, padded.d_qk_scale_padded,
                                        k_scale_strides.data(), kv_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_scale_padded, padded.d_v_padded,
                                        v_scale_strides.data(), kv_scale_inv_format);
        descale_q =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_q")
                                  .set_dim({b, h, padded.s_q_padded, padded.d_qk_scale_padded})
                                  .set_stride(q_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_k =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_k")
                                  .set_dim({b, hg, padded.s_kv_padded, padded.d_qk_scale_padded})
                                  .set_stride(k_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_v =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_v")
                                  .set_dim({b, hg, padded.s_kv_scale_padded, padded.d_v_padded})
                                  .set_stride(v_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
      }

      fe::graph::SDPA_fp8_attributes sdpa_options;
      sdpa_options = fe::graph::SDPA_fp8_attributes()
                         .set_name("sdpa_fp8")
                         .set_generate_stats(true)
                         .set_causal_mask(is_causal)
                         .set_attn_scale(attn_scale);

      fe::DiagonalAlignment_t const& diagonal_alignment =
          bottom_right_diagonal ? fe::DiagonalAlignment_t::BOTTOM_RIGHT
                                : fe::DiagonalAlignment_t::TOP_LEFT;
      sdpa_options.set_diagonal_alignment(diagonal_alignment);

      if (cudnn_runtime_version >= 92100) {
        if (window_size_left != -1) {
          sdpa_options.set_diagonal_band_left_bound(window_size_left + 1);
        }
        if (window_size_right != -1) {
          sdpa_options.set_diagonal_band_right_bound(window_size_right);
        }
      }

      // sdpa_options.set_alibi_mask(is_alibi);
      // if (is_bias) {
      //     bias = mha_graph->tensor(fe::graph::Tensor_attributes()
      //                     .set_name("bias")
      //                     .set_dim({bias_b, bias_h, bias_sq, bias_skv})
      //                     .set_stride({bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1}));
      //     sdpa_options.set_bias(bias);
      // }

      if (is_padding) {
        seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("seq_q")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
        seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_kv")
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
        sdpa_options.set_padding_mask(is_padding).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
      }

      if (is_dropout) {
        dropout_seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                             .set_name("Seed")
                                             .set_dim({1, 1, 1, 1})
                                             .set_stride({1, 1, 1, 1})
                                             .set_data_type(fe::DataType_t::INT64));
        dropout_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                               .set_name("Offset")
                                               .set_dim({1, 1, 1, 1})
                                               .set_stride({1, 1, 1, 1})
                                               .set_data_type(fe::DataType_t::INT64));
        sdpa_options.set_dropout(dropout_probability, dropout_seed, dropout_offset);
      }

      if (is_softmax_offset) {
        softmax_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                               .set_name("softmax_offset")
                                               .set_dim({1, h, 1, 1})
                                               .set_stride({h, 1, 1, 1})
                                               .set_data_type(fe::DataType_t::FLOAT));
        sdpa_options.set_sink_token(softmax_offset);
      }

      std::shared_ptr<fe::graph::Tensor_attributes> O, Stats, amax_s, amax_o;
      if (is_delayed_scaling || is_current_scaling) {
        auto outputs = mha_graph->sdpa_fp8(Q, K, V, descale_q, descale_k, descale_v, descale_s,
                                           scale_s, scale_o, sdpa_options);
        O = outputs[0];
        Stats = outputs[1];
        amax_s = outputs[2];
        amax_o = outputs[3];
        amax_s->set_output(true)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT);
      } else if (is_mxfp8) {
        auto outputs = mha_graph->sdpa_fp8(Q, K, V, descale_q, descale_k, descale_v, sdpa_options);
        O = outputs[0];
        Stats = outputs[1];
        amax_o = outputs[2];
      }

      std::vector<int64_t> o_strides(4);
      generateMatrixStridesWithFormat(b, h, s_q, d_v, o_strides.data(), o_format);
      O->set_output(true)
          .set_dim({b, h, s_q, d_v})
          .set_stride(o_strides)
          .set_data_type(o_tensor_type);
      if (is_ragged) {
        O->set_ragged_offset(offset_o);
      }
      amax_o->set_output(!is_mxfp8)
          .set_dim({1, 1, 1, 1})
          .set_stride({1, 1, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT);

      Stats->set_output(true)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({b, h, s_q, 1})
          .set_stride(is_ragged ? std::vector<int64_t>{h * s_q, 1, h, 1}
                                : std::vector<int64_t>{h * s_q, s_q, 1, 1});
      if (is_ragged) {
        // The Stats ragged-offset multiplier is h, not h*d -- Stats is one value per
        // (token, head). Getting this wrong is invisible in the forward output but corrupts the
        // backward, which consumes Stats as LSE.
        Stats->set_ragged_offset(offset_stats);
      }

      std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // K
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // V
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_k
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_v
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_s
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_s
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_o
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // O
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_s
                 std::shared_ptr<fe::graph::Tensor_attributes>>  // amax_o
          key_tensors_tuple =
              is_mxfp8 ? std::make_tuple(Q, K, V, descale_q, descale_k, descale_v, nullptr, nullptr,
                                         nullptr, attn_scale, O, nullptr, amax_o)
                       : std::make_tuple(Q, K, V, descale_q, descale_k, descale_v, descale_s,
                                         scale_s, scale_o, attn_scale, O, amax_s, amax_o);
      auto Stats_tuple = std::make_tuple(Stats);
      auto bias_tuple = is_bias ? std::make_tuple(bias) : std::make_tuple(nullptr);
      auto softmax_offset_tuple =
          is_softmax_offset ? std::make_tuple(softmax_offset) : std::make_tuple(nullptr);
      auto padding_tuple =
          is_padding ? std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);
      auto ragged_tuple =
          is_ragged ? std::make_tuple(offset_q, offset_k, offset_v, offset_o, offset_stats)
                    : std::make_tuple(nullptr, nullptr, nullptr, nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));
      auto return_tuple =
          std::tuple_cat(std::make_tuple(mha_graph), key_tensors_tuple, Stats_tuple, bias_tuple,
                         softmax_offset_tuple, padding_tuple, dropout_tuple, ragged_tuple);
      cache.insert({descriptor, return_tuple});
      if (fp8_cache_stats_enabled())
        fp8_cache_stats.fprop_entries.store(cache.size(), std::memory_order_relaxed);

      return return_tuple;
    };

    auto [mha_graph, Q, K, V, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
          attn_scale, O, amax_s, amax_o, Stats, bias, softmax_offset, seq_q, seq_kv, dropout_seed,
          dropout_offset, offset_q, offset_k, offset_v, offset_o, offset_stats] =
        get_graph(sdpa_fp8_fprop_cache, descriptor);

    // Ragged input appends int64 ragged offsets after these buffers, so both the plan size and each
    // sequence array have to be individually aligned or the offsets start misaligned. Non-ragged
    // appends nothing after the pair and keeps its original layout exactly, so this change cannot
    // move a byte on the BSHD, SBHD or BHSD paths.
    const size_t plan_workspace_size =
        is_ragged ? alignTo<16>(mha_graph->get_workspace_size()) : mha_graph->get_workspace_size();

    // Exit to request upper level API to allocate memory if needed.
    const size_t num_bytes_per_seqlen =
        is_ragged ? alignTo<16>(b * sizeof(int32_t)) : b * sizeof(int32_t);
    const size_t actual_seqlen_workspace_size =
        is_ragged ? 2 * num_bytes_per_seqlen : alignTo<16>(2 * b * sizeof(int32_t));
    const size_t num_bytes_per_ragged_offset =
        alignTo<16>(((b + 1) * typeToNumBits(ragged_offset_type)) / 8);
    // Q, K, V, O, Stats
    const size_t ragged_offsets_workspace_size = is_ragged ? 5 * num_bytes_per_ragged_offset : 0;
    if (workspace == nullptr) {
      *workspace_size =
          plan_workspace_size + actual_seqlen_workspace_size + ragged_offsets_workspace_size;
      return;
    }

    // cuDNN stream check needs to be moved here to support dummy kernel calls with
    // null streams for sizing the cuDNN workspace.
    NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ},
        {K, devPtrK},
        {V, devPtrV},
        {descale_q, devPtrDescaleQ},
        {descale_k, devPtrDescaleK},
        {descale_v, devPtrDescaleV},
        {attn_scale, &scaling_factor},
        {O, devPtrO},
        {Stats, devPtrM}};

    if (is_delayed_scaling) {
      variant_pack[scale_o] = devPtrScaleO;
    }
    if (is_delayed_scaling || is_current_scaling) {
      variant_pack[descale_s] = devPtrDescaleS;
      variant_pack[scale_s] = devPtrScaleS;
      variant_pack[amax_s] = devPtrAmaxS;
      variant_pack[amax_o] = devPtrAmaxO;
    }

    /* if (is_bias) {
       variant_pack[bias] = devPtrBias;
    } */

    if (is_padding) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
      void* devActualSeqlenQ = static_cast<int8_t*>(workspace) + plan_workspace_size;
      void* devActualSeqlenKV = static_cast<int8_t*>(devActualSeqlenQ) + num_bytes_per_seqlen;
      // (actual_b, b): read only the entries cu_seqlens actually has, and zero-fill any remainder
      // out to the graph's batch dimension. b is not currently bucketed, so the two are equal and
      // the second argument has no effect today; passing both keeps this correct if a batch
      // capacity is ever introduced.
      cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
          actual_b, b, static_cast<const int32_t*>(devPtrcuSeqlensQ),
          static_cast<const int32_t*>(devPtrcuSeqlensKV), static_cast<int32_t*>(devActualSeqlenQ),
          static_cast<int32_t*>(devActualSeqlenKV));
      NVTE_CHECK_CUDA(cudaGetLastError());
      variant_pack[seq_q] = devActualSeqlenQ;
      variant_pack[seq_kv] = devActualSeqlenKV;

      if (is_ragged) {
        // Element offsets into the packed buffers: offsets_x[i] = mult_x * cu_seqlens[i].
        // The multipliers are layout-specific (h*d for separate Q/K/V, 3*h*d for t3hd/th3d,
        // 2*h_kv*d for the KV-packed layouts, and h -- not h*d -- for Stats), which is exactly
        // what get_ragged_offset_multipliers encodes for the F16 path.
        //
        // This kernel writes b+1 entries, one more than the seqlen conversion above, so it needs
        // its own launch extent -- (b + nthreads) / nthreads, matching
        // fused_attn_f16_arbitrary_seqlen.cu:498. Sharing the seqlen grid leaves the terminal
        // offset unwritten whenever b is an exact multiple of the block size, so a packed batch of
        // exactly 128, 256 or 512 documents would address its last document from an uninitialised
        // offset.
        const size_t offsets_grid = (b + nthreads_per_block) / nthreads_per_block;
        int8_t* devOffsets =
            static_cast<int8_t*>(workspace) + plan_workspace_size + actual_seqlen_workspace_size;
        void* devOffsetsQ = devOffsets;
        void* devOffsetsK = devOffsets + num_bytes_per_ragged_offset;
        void* devOffsetsV = devOffsets + 2 * num_bytes_per_ragged_offset;
        void* devOffsetsO = devOffsets + 3 * num_bytes_per_ragged_offset;
        void* devOffsetsS = devOffsets + 4 * num_bytes_per_ragged_offset;

        const NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
        // Ragged base offsets from cu_seqlens_*_padded (physical slots), so a document is addressed
        // at its padded base. The kernel derives the per-tensor multipliers from layout_group/h/hg/d
        // -- h*d for separate Q/K/V, 3*h*d for t3hd/th3d, 2*hg*d for the KV-packed layouts, and h
        // (not h*d) for Stats. Matches the F16 path (fused_attn_f16_arbitrary_seqlen.cu:521).
        cu_seqlens_padded_to_offsets<<<offsets_grid, nthreads_per_block, 0, stream>>>(
            layout_group, actual_b, b, h, hg, d_qk, d_v,
            static_cast<const int32_t*>(devPtrcuSeqlensQPadded),
            static_cast<const int32_t*>(devPtrcuSeqlensKVPadded), ragged_offset_type, devOffsetsQ,
            devOffsetsK, devOffsetsV, devOffsetsO, devOffsetsS);
        NVTE_CHECK_CUDA(cudaGetLastError());

        variant_pack[offset_q] = devOffsetsQ;
        variant_pack[offset_k] = devOffsetsK;
        variant_pack[offset_v] = devOffsetsV;
        variant_pack[offset_o] = devOffsetsO;
        variant_pack[offset_stats] = devOffsetsS;
      }
    }

    if (is_dropout) {
      variant_pack[dropout_seed] = devPtrDropoutSeed;
      variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

    if (is_softmax_offset) {
      variant_pack[softmax_offset] = devPtrSoftmaxOffset;
    }

    NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
  } catch (cudnn_frontend::cudnnException& e) {
    NVTE_ERROR(e.what());
  }
}

// fused attention BWD FP8 with FE 1.0+
void fused_attn_fp8_bwd_impl(
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d_qk, int64_t d_v,
    float scaling_factor, float dropout_probability, NVTE_QKV_Layout qkv_layout,
    NVTE_QKV_Format o_format, NVTE_QKV_Format do_format, NVTE_QKV_Layout dqkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, NVTE_Softmax_Type softmax_type,
    int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal,
    bool deterministic, void* devPtrQ, void* devPtrK, void* devPtrV, void* devPtrM, void* devPtrO,
    void* devPtrdO, void* devPtrSoftmaxOffset, void* devPtrdQ, void* devPtrdK, void* devPtrdV,
    void* devPtrdSoftmaxOffset, void* devPtrDescaleQ, void* devPtrDescaleK, void* devPtrDescaleV,
    void* devPtrDescaleO, void* devPtrDescaledO, void* devPtrDescaleS, void* devPtrDescaledP,
    void* devPtrScaleS, void* devPtrScaledP, void* devPtrScaledQ, void* devPtrScaledK,
    void* devPtrScaledV, void* devPtrAmaxdP, void* devPtrAmaxdQ, void* devPtrAmaxdK,
    void* devPtrAmaxdV, void* devPtrQ_t, void* devPtrK_t, void* devPtrdO_f16, void* devPtrdO_t,
    void* devPtrDescaleQ_t, void* devPtrDescaleK_t, void* devPtrDescaledO_t, void* devPtrcuSeqlensQ,
    void* devPtrcuSeqlensKV, void* devPtrcuSeqlensQPadded, void* devPtrcuSeqlensKVPadded,
    void* devPtrDropoutSeed, void* devPtrDropoutOffset, cudnn_frontend::DataType_t qkv_tensor_type,
    cudnn_frontend::DataType_t o_tensor_type, cudnn_frontend::DataType_t do_tensor_type,
    cudnn_frontend::DataType_t dqkv_tensor_type, NVTEScalingMode scaling_mode,
    NVTE_QKV_Format qkv_scale_inv_format, NVTE_QKV_Format do_scale_inv_format, void* workspace,
    size_t* workspace_size, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  const auto cudnn_runtime_version = cudnnGetVersion();
  bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
  bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
  bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
                    (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_dropout = (dropout_probability != 0.0f);
  bool is_softmax_offset = (softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX);
  auto bias_b = b;
  auto bias_h = h;
  auto bias_sq = s_q;
  auto bias_skv = s_kv;
  NVTE_CHECK(~is_bias, "FP8 fused attention does not support pre/post_scale_bias yet!");
  NVTE_CHECK(~is_alibi, "FP8 fused attention does not support ALiBi yet!");
  bool is_delayed_scaling = (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) &&
                            (dqkv_tensor_type == cudnn_frontend::DataType_t::FP8_E4M3 ||
                             dqkv_tensor_type == cudnn_frontend::DataType_t::FP8_E5M2);
  bool is_current_scaling = (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) &&
                            (dqkv_tensor_type == cudnn_frontend::DataType_t::HALF ||
                             dqkv_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);
  bool is_mxfp8 = (scaling_mode == NVTE_MXFP8_1D_SCALING) &&
                  (dqkv_tensor_type == cudnn_frontend::DataType_t::HALF ||
                   dqkv_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);
  NVTE_CHECK(
      is_delayed_scaling || is_current_scaling || is_mxfp8,
      "FP8 fused attention only supports FP8DelayedScaling or FP8CurrentScaling or MXFP8 recipes!");
  NVTE_CHECK(!is_mxfp8 || cudnn_runtime_version >= 92100,
             "MXFP8 fused attention requires cuDNN 9.21.0 or later!");

  bool is_O_in_F16 = (o_tensor_type == cudnn_frontend::DataType_t::HALF ||
                      o_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);

  // THD (packed varlen), mirroring fused_attn_fp8_fwd_impl. Gap-free only: cu_seqlens doubles as
  // cu_seqlens_padded, so element offsets come from the cu_seqlens this entry point already
  // receives.
  //
  // The backward needs two sets of offsets, not one. Reads (Q/K/V/O/Stats) are addressed by
  // qkv_layout; writes (dQ/dK/dV) by dqkv_layout. Those layouts can differ -- a t3hd forward can
  // produce separate thd gradients -- and the multipliers differ with them (3*h*d vs h*d). When
  // they happen to match, the second set is identical and costs one extra launch plus five small
  // buffers, which is cheaper than being subtly wrong when they do not.
  const bool is_ragged = (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD);
  NVTE_CHECK(!is_ragged || is_padding,
             "FP8 fused attention with THD requires a padding or padding_causal mask!");
  NVTE_CHECK(!is_ragged || nvte_get_qkv_format(dqkv_layout) == NVTE_QKV_Format::NVTE_THD,
             "FP8 fused attention with THD requires THD gradients (dqkv_layout must be THD)!");

  // Same quantization as the forward, and it must agree with it: the backward keys its own graph
  // cache on the same varying dimensions. See fused_attn_fp8_fwd_impl.
  const int64_t actual_b = b;
  if (is_ragged) {
    // Must bucket identically to the forward: the two graphs key on the same dimensions, and the
    // backward workspace responds to them the same way. See fused_attn_fp8_fwd_impl.
    NVTE_CHECK(actual_b > 0, "FP8 THD attention requires a positive batch size.");
    const int64_t bucketed_s_q = static_cast<int64_t>(fp8_thd_sequence_bucket(s_q));
    const int64_t bucketed_s_kv = static_cast<int64_t>(fp8_thd_sequence_bucket(s_kv));
    NVTE_CHECK(bucketed_s_q >= s_q, "FP8 THD query bucket ", bucketed_s_q,
               " is below the longest query sequence ", s_q, ".");
    NVTE_CHECK(bucketed_s_kv >= s_kv, "FP8 THD key/value bucket ", bucketed_s_kv,
               " is below the longest key/value sequence ", s_kv, ".");
    s_q = bucketed_s_q;
    s_kv = bucketed_s_kv;
    bias_sq = s_q;
    bias_skv = s_kv;
  }

  const DType ragged_offset_type = cudnn_runtime_version >= 90500 ? DType::kInt64 : DType::kInt32;

  try {
    FADescriptor_v1 descriptor{b,
                               h,
                               hg,
                               s_q,
                               s_kv,
                               d_qk,
                               d_v,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               bias_b,
                               bias_h,
                               bias_sq,
                               bias_skv,
                               scaling_factor,
                               true,
                               dropout_probability,
                               qkv_layout,
                               o_format,
                               do_format,
                               dqkv_layout,
                               qkv_scale_inv_format,
                               do_scale_inv_format,
                               bias_type,
                               mask_type,
                               softmax_type,
                               window_size_left,
                               window_size_right,
                               bottom_right_diagonal,
                               deterministic,
                               qkv_tensor_type,
                               o_tensor_type,
                               do_tensor_type,
                               dqkv_tensor_type,
                               false};

    namespace fe = cudnn_frontend;
    using graph_and_tensors =
        std::tuple<std::shared_ptr<fe::graph::Graph>,
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Q_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // K
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // K_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // V
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // O
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dO
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dO_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dO_f16
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // attn_scale
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_q_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_k_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_dO
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_dO_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_dP
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_dQ
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_dK
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_dV
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_dP
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dQ
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dK
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dV
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_dQ
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_dK
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_dV
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_dP
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // bias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dBias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // softmax_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // d_softmax_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_do
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_dq
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_dk
                   std::shared_ptr<fe::graph::Tensor_attributes>>;  // offset_dv

    using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
    static thread_local CacheType sdpa_fp8_bprop_cache;

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph = [&](CacheType& cache, const FADescriptor_v1& descriptor) -> graph_and_tensors {
      // if hit, return
      if (fp8_cache_stats_enabled())
        fp8_cache_stats.bprop_lookups.fetch_add(1, std::memory_order_relaxed);
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        if (fp8_cache_stats_enabled())
          fp8_cache_stats.bprop_hits.fetch_add(1, std::memory_order_relaxed);
        auto graph = it->second;
        return graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();

      mha_graph->set_io_data_type(qkv_tensor_type)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> Q, Q_t, K, K_t, V, O, dO, dO_t, dO_f16, Stats,
          attn_scale;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_q, descale_q_t, descale_k, descale_k_t,
          descale_v;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_s, descale_o;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_dP, descale_dO, descale_dO_t;
      std::shared_ptr<fe::graph::Tensor_attributes> scale_s, scale_dP;
      std::shared_ptr<fe::graph::Tensor_attributes> scale_dQ, scale_dK, scale_dV;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, dBias, softmax_offset, d_softmax_offset;
      std::shared_ptr<fe::graph::Tensor_attributes> seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;
      // Reads, addressed by qkv_layout.
      std::shared_ptr<fe::graph::Tensor_attributes> offset_q, offset_k, offset_v, offset_o,
          offset_stats;
      // Gradients, addressed by dqkv_layout.
      std::shared_ptr<fe::graph::Tensor_attributes> offset_do, offset_dq, offset_dk, offset_dv;

      // Q, K, V, O, dO, stats, attn_scale
      std::vector<int64_t> q_strides(4), k_strides(4), v_strides(4), o_strides(4), dO_strides(4);
      generateMatrixStridesWithLayout(b, h, hg, s_q, s_kv, d_qk, d_v, q_strides.data(),
                                      k_strides.data(), v_strides.data(), qkv_layout);
      generateMatrixStridesWithFormat(b, h, s_q, d_v, o_strides.data(), o_format);
      generateMatrixStridesWithFormat(b, h, s_q, d_v, dO_strides.data(), do_format);
      Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q")
                                .set_dim({b, h, s_q, d_qk})
                                .set_stride(q_strides)
                                .set_data_type(qkv_tensor_type));
      K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K")
                                .set_dim({b, hg, s_kv, d_qk})
                                .set_stride(k_strides)
                                .set_data_type(qkv_tensor_type));
      V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V")
                                .set_dim({b, hg, s_kv, d_v})
                                .set_stride(v_strides)
                                .set_data_type(qkv_tensor_type));
      O = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("O")
                                .set_dim({b, h, s_q, d_v})
                                .set_stride(o_strides)
                                .set_data_type(o_tensor_type));
      dO = mha_graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("dO")
                                 .set_dim({b, h, s_q, d_v})
                                 .set_stride(dO_strides)
                                 .set_data_type(do_tensor_type));
      Stats =
          mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Stats")
                                .set_dim({b, h, s_q, 1})
                                // Packed Stats is [t, h, 1], so the sequence axis strides by
                                // h and the head axis by 1 -- the transpose of the dense case.
                                .set_stride(is_ragged ? std::vector<int64_t>{h * s_q, 1, h, 1}
                                                      : std::vector<int64_t>{h * s_q, s_q, 1, 1})
                                .set_data_type(fe::DataType_t::FLOAT));
      // THD: dense {b,h,s,d} dims with BSHD-style strides, plus a per-batch ragged offset that
      // displaces each sequence's base pointer into the packed buffer. The Stats multiplier is
      // h, not h*d -- Stats holds one value per (token, head), not per element.
      if (is_ragged) {
        auto make_offset = [&](const char* name) {
          return mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name(name)
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        };
        offset_q = make_offset("offset_q");
        offset_k = make_offset("offset_k");
        offset_v = make_offset("offset_v");
        offset_o = make_offset("offset_o");
        offset_stats = make_offset("offset_stats");
        offset_do = make_offset("offset_do");
        offset_dq = make_offset("offset_dq");
        offset_dk = make_offset("offset_dk");
        offset_dv = make_offset("offset_dv");
        Q->set_ragged_offset(offset_q);
        K->set_ragged_offset(offset_k);
        V->set_ragged_offset(offset_v);
        O->set_ragged_offset(offset_o);
        dO->set_ragged_offset(offset_do);
        Stats->set_ragged_offset(offset_stats);
      }
      attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("attn_scale")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_is_pass_by_value(true)
                                         .set_data_type(fe::DataType_t::FLOAT));

      // Descale_q, Descale_k, Descale_v, Descale_s, Scale_s, Descale_dP, Scale_dP, Descale_o, Descale_dO, Scale_dQ, Scale_dK, Scale_dV
      if (is_delayed_scaling || is_current_scaling) {
        descale_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("Descale_q")
                                          .set_dim({1, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::FLOAT));
        descale_k = mha_graph->tensor_like(descale_q, "Descale_q");
        descale_v = mha_graph->tensor_like(descale_q, "Descale_v");
        descale_s = mha_graph->tensor_like(descale_q, "Descale_s");
        scale_s = mha_graph->tensor_like(descale_q, "Scale_s");
        descale_dP = mha_graph->tensor_like(descale_q, "Descale_dP");
        scale_dP = mha_graph->tensor_like(descale_q, "Scale_dP");
        if (is_current_scaling && is_O_in_F16) {
          descale_o = mha_graph->tensor(1.0f);
        } else {
          descale_o = mha_graph->tensor_like(descale_q, "Descale_O");
        }
        descale_dO = mha_graph->tensor_like(descale_q, "Descale_dO");
        if (is_delayed_scaling) {
          scale_dQ = mha_graph->tensor_like(descale_q, "Scale_dQ");
          scale_dK = mha_graph->tensor_like(descale_q, "Scale_dK");
          scale_dV = mha_graph->tensor_like(descale_q, "Scale_dV");
        }
        if (is_current_scaling) {
          scale_dQ = mha_graph->tensor(1.0f);
          scale_dK = mha_graph->tensor(1.0f);
          scale_dV = mha_graph->tensor(1.0f);
        }
      } else if (is_mxfp8) {
        NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
        NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
        NVTE_QKV_Format q_scale_inv_format =
            (qkv_scale_inv_format != NVTE_QKV_Format_NOT_SET) ? qkv_scale_inv_format : q_format;
        NVTE_QKV_Format kv_scale_inv_format =
            (qkv_scale_inv_format != NVTE_QKV_Format_NOT_SET) ? qkv_scale_inv_format : kv_format;
        NVTE_QKV_Format do_scale_format_ =
            (do_scale_inv_format != NVTE_QKV_Format_NOT_SET) ? do_scale_inv_format : do_format;
        // Q_t, K_t, dO_t, dO_f16
        std::vector<int64_t> q_t_strides(4), k_t_strides(4), dO_t_strides(4);
        generateMatrixStridesWithFormat(b, h, s_q, d_qk, q_t_strides.data(), q_format);
        generateMatrixStridesWithFormat(b, hg, s_kv, d_qk, k_t_strides.data(), kv_format);
        generateMatrixStridesWithFormat(b, h, s_q, d_v, dO_t_strides.data(), do_format);
        Q_t = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Q_t")
                                    .set_dim({b, h, s_q, d_qk})
                                    .set_stride(q_t_strides)
                                    .set_data_type(qkv_tensor_type));
        K_t = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("K_t")
                                    .set_dim({b, hg, s_kv, d_qk})
                                    .set_stride(k_t_strides)
                                    .set_data_type(qkv_tensor_type));
        dO_t = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("dO_t")
                                     .set_dim({b, h, s_q, d_v})
                                     .set_stride(dO_t_strides)
                                     .set_data_type(do_tensor_type));
        dO_f16 = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("dO_f16")
                                       .set_dim({b, h, s_q, d_v})
                                       .set_stride(dO_strides)
                                       .set_data_type(o_tensor_type));
        // Descale_q, Descale_q_t, Descale_k, Descale_k_t, Descale_v, Descale_dO, Descale_dO_t
        auto padded = pad_s_d_for_mxfp8(s_q, s_kv, d_qk, d_v);
        std::vector<int64_t> q_scale_strides(4), q_t_scale_strides(4), k_scale_strides(4),
            k_t_scale_strides(4), v_scale_strides(4), dO_scale_strides(4), dO_t_scale_strides(4);
        generateMatrixStridesWithFormat(b, h, padded.s_q_padded, padded.d_qk_scale_padded,
                                        q_scale_strides.data(), q_scale_inv_format);
        generateMatrixStridesWithFormat(b, h, padded.s_q_scale_padded, padded.d_qk_padded,
                                        q_t_scale_strides.data(), q_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_padded, padded.d_qk_scale_padded,
                                        k_scale_strides.data(), kv_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_scale_padded, padded.d_qk_padded,
                                        k_t_scale_strides.data(), kv_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_padded, padded.d_v_scale_padded,
                                        v_scale_strides.data(), kv_scale_inv_format);
        generateMatrixStridesWithFormat(b, h, padded.s_q_padded, padded.d_v_scale_padded,
                                        dO_scale_strides.data(), do_scale_format_);
        generateMatrixStridesWithFormat(b, h, padded.s_q_scale_padded, padded.d_v_padded,
                                        dO_t_scale_strides.data(), do_scale_format_);
        descale_q =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_q")
                                  .set_dim({b, h, padded.s_q_padded, padded.d_qk_scale_padded})
                                  .set_stride(q_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_q_t =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_q_t")
                                  .set_dim({b, h, padded.s_q_scale_padded, padded.d_qk_padded})
                                  .set_stride(q_t_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_k =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_k")
                                  .set_dim({b, hg, padded.s_kv_padded, padded.d_qk_scale_padded})
                                  .set_stride(k_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_k_t =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_k_t")
                                  .set_dim({b, hg, padded.s_kv_scale_padded, padded.d_qk_padded})
                                  .set_stride(k_t_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_v =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_v")
                                  .set_dim({b, hg, padded.s_kv_padded, padded.d_v_scale_padded})
                                  .set_stride(v_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_dO =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_dO")
                                  .set_dim({b, h, padded.s_q_padded, padded.d_v_scale_padded})
                                  .set_stride(dO_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_dO_t =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_dO_t")
                                  .set_dim({b, h, padded.s_q_scale_padded, padded.d_v_padded})
                                  .set_stride(dO_t_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
      }

      fe::graph::SDPA_fp8_backward_attributes sdpa_backward_options;
      sdpa_backward_options = fe::graph::SDPA_fp8_backward_attributes()
                                  .set_name("sdpa_fp8_backward")
                                  .set_causal_mask(is_causal)
                                  .set_attn_scale(attn_scale);

      fe::DiagonalAlignment_t const& diagonal_alignment =
          bottom_right_diagonal ? fe::DiagonalAlignment_t::BOTTOM_RIGHT
                                : fe::DiagonalAlignment_t::TOP_LEFT;
      sdpa_backward_options.set_diagonal_alignment(diagonal_alignment);

      if (cudnn_runtime_version >= 92100) {
        if (window_size_left != -1) {
          sdpa_backward_options.set_diagonal_band_left_bound(window_size_left + 1);
        }
        if (window_size_right != -1) {
          sdpa_backward_options.set_diagonal_band_right_bound(window_size_right);
        }
      }

      // sdpa_backward_options.set_alibi_mask(is_alibi);

      // if (is_bias) {
      //     bias = mha_graph->tensor(fe::graph::Tensor_attributes()
      //                     .set_name("bias")
      //                     .set_dim({bias_b, bias_h, bias_sq, bias_skv})
      //                     .set_stride({bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1}));
      //     dBias = mha_graph->tensor(fe::graph::Tensor_attributes()
      //                     .set_name("dBias")
      //                     .set_dim({bias_b, bias_h, bias_sq, bias_skv})
      //                     .set_stride({bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1}));
      //     sdpa_backward_options.set_bias(bias);
      // bias shapes [1, 1, s, s], [b, 1, s, s], [b, h, s, s], [1, h, s, s] are supported for dbias calculation
      // bias shape [1, 1, 1, s] is not supported for dbias calculation as of cuDNN 9.18
      // if (!((bias_b == 1) && (bias_h == 1) && (bias_sq == 1))) {
      //    sdpa_backward_options.set_dbias(dBias);
      //  }
      // }

      if (cudnn_runtime_version >= 91900) {
        sdpa_backward_options.set_deterministic_algorithm(deterministic);
      }

      if (is_padding) {
        seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("seq_q")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
        seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_kv")
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
        sdpa_backward_options.set_padding_mask(is_padding)
            .set_seq_len_q(seq_q)
            .set_seq_len_kv(seq_kv);
      }

      if (is_dropout) {
        dropout_seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                             .set_name("Seed")
                                             .set_dim({1, 1, 1, 1})
                                             .set_stride({1, 1, 1, 1})
                                             .set_data_type(fe::DataType_t::INT64));
        dropout_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                               .set_name("Offset")
                                               .set_dim({1, 1, 1, 1})
                                               .set_stride({1, 1, 1, 1})
                                               .set_data_type(fe::DataType_t::INT64));
        sdpa_backward_options.set_dropout(dropout_probability, dropout_seed, dropout_offset);
      }

      if (is_softmax_offset) {
        softmax_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                               .set_name("softmax_offset")
                                               .set_dim({1, h, 1, 1})
                                               .set_stride({h, 1, 1, 1})
                                               .set_data_type(fe::DataType_t::FLOAT));
        sdpa_backward_options.set_sink_token(softmax_offset);
        d_softmax_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                                 .set_name("d_softmax_offset")
                                                 .set_dim({1, h, 1, 1})
                                                 .set_stride({h, 1, 1, 1})
                                                 .set_data_type(fe::DataType_t::FLOAT));
        sdpa_backward_options.set_dsink_token(d_softmax_offset);
      }

      std::shared_ptr<fe::graph::Tensor_attributes> dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP;
      if (is_delayed_scaling || is_current_scaling) {
        std::tie(dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP) =
            std::apply([](const auto&... elems) { return std::make_tuple(elems...); },
                       mha_graph->sdpa_fp8_backward(Q, K, V, O, dO, Stats, descale_q, descale_k,
                                                    descale_v, descale_o, descale_dO, descale_s,
                                                    descale_dP, scale_s, scale_dQ, scale_dK,
                                                    scale_dV, scale_dP, sdpa_backward_options));
      } else if (is_mxfp8) {
        std::tie(dQ, dK, dV, amax_dQ, amax_dK, amax_dV) = std::apply(
            [](const auto&... elems) { return std::make_tuple(elems...); },
            mha_graph->sdpa_fp8_backward(Q, Q_t, K, K_t, V, O, dO_f16, dO, dO_t, Stats, descale_q,
                                         descale_q_t, descale_k, descale_k_t, descale_v, descale_dO,
                                         descale_dO_t, sdpa_backward_options));
      }
      std::vector<int64_t> dq_strides(4), dk_strides(4), dv_strides(4);
      generateMatrixStridesWithLayout(b, h, hg, s_q, s_kv, d_qk, d_v, dq_strides.data(),
                                      dk_strides.data(), dv_strides.data(), dqkv_layout);
      dQ->set_output(true)
          .set_dim({b, h, s_q, d_qk})
          .set_stride(dq_strides)
          .set_data_type(dqkv_tensor_type);
      dK->set_output(true)
          .set_dim({b, hg, s_kv, d_qk})
          .set_stride(dk_strides)
          .set_data_type(dqkv_tensor_type);
      dV->set_output(true)
          .set_dim({b, hg, s_kv, d_v})
          .set_stride(dv_strides)
          .set_data_type(dqkv_tensor_type);
      // Gradient writes use the dqkv_layout offset set, which is computed from dqkv_layout and so
      // carries that layout's multipliers even when it differs from qkv_layout.
      if (is_ragged) {
        dQ->set_ragged_offset(offset_dq);
        dK->set_ragged_offset(offset_dk);
        dV->set_ragged_offset(offset_dv);
      }
      amax_dQ->set_output(!is_mxfp8)
          .set_dim({1, 1, 1, 1})
          .set_stride({1, 1, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT);
      amax_dK->set_output(!is_mxfp8)
          .set_dim({1, 1, 1, 1})
          .set_stride({1, 1, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT);
      amax_dV->set_output(!is_mxfp8)
          .set_dim({1, 1, 1, 1})
          .set_stride({1, 1, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT);
      if (is_delayed_scaling || is_current_scaling) {
        amax_dP->set_output(true)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT);
      }

      std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // K
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // V
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // O
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // Stats
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dO
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_k
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_v
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_o
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_dO
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_s
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_dP
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dQ
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dK
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dV
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_s
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dP
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dQ
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dK
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dV
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dQ
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dK
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dV
                 std::shared_ptr<fe::graph::Tensor_attributes>>  // amax_dP
          key_tensors_tuple = std::make_tuple(
              Q, K, V, O, Stats, dO, attn_scale, descale_q, descale_k, descale_v, descale_o,
              descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
              dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP);
      auto mxfp8_tensors_tuple =
          is_mxfp8 ? std::make_tuple(Q_t, K_t, dO_f16, dO_t, descale_q_t, descale_k_t, descale_dO_t)
                   : std::make_tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
      auto bias_tuple = is_bias ? std::make_tuple(bias, dBias) : std::make_tuple(nullptr, nullptr);
      auto softmax_offset_tuple = is_softmax_offset
                                      ? std::make_tuple(softmax_offset, d_softmax_offset)
                                      : std::make_tuple(nullptr, nullptr);
      auto padding_tuple =
          is_padding ? std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);
      auto ragged_tuple =
          is_ragged ? std::make_tuple(offset_q, offset_k, offset_v, offset_o, offset_stats,
                                      offset_do, offset_dq, offset_dk, offset_dv)
                    : std::make_tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

      auto return_tuple = std::tuple_cat(std::make_tuple(mha_graph), key_tensors_tuple,
                                         mxfp8_tensors_tuple, bias_tuple, softmax_offset_tuple,
                                         padding_tuple, dropout_tuple, ragged_tuple);
      cache.insert({descriptor, return_tuple});
      if (fp8_cache_stats_enabled())
        fp8_cache_stats.bprop_entries.store(cache.size(), std::memory_order_relaxed);

      return return_tuple;
    };
    auto [mha_graph, Q, K, V, O, Stats, dO, attn_scale, descale_q, descale_k, descale_v, descale_o,
          descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP, dQ,
          dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP, Q_t, K_t, dO_f16, dO_t, descale_q_t,
          descale_k_t, descale_dO_t, bias, dBias, softmax_offset, d_softmax_offset, seq_q, seq_kv,
          dropout_seed, dropout_offset, offset_q, offset_k, offset_v, offset_o, offset_stats,
          offset_do, offset_dq, offset_dk, offset_dv] = get_graph(sdpa_fp8_bprop_cache, descriptor);

    // See fused_attn_fp8_fwd_impl. Ragged aligns the plan and each sequence array because int64
    // ragged offsets follow them; non-ragged keeps its original layout, which had nothing after the
    // sequence pair and so never needed the padding.
    const size_t plan_workspace_size =
        is_ragged ? alignTo<16>(mha_graph->get_workspace_size()) : mha_graph->get_workspace_size();

    // Exit to request upper level API to allocate memory if needed.
    const size_t num_bytes_per_seqlen =
        is_ragged ? alignTo<16>(b * sizeof(int32_t)) : b * sizeof(int32_t);
    const size_t actual_seqlen_workspace_size =
        is_ragged ? 2 * num_bytes_per_seqlen : 2 * b * sizeof(int32_t);
    const size_t num_bytes_per_ragged_offset =
        alignTo<16>(((b + 1) * typeToNumBits(ragged_offset_type)) / 8);
    // Two sets of five: {Q,K,V,O,Stats} from qkv_layout and {dQ,dK,dV,dO,unused} from
    // dqkv_layout. cu_seqlens_padded_to_offsets always writes five outputs, so the second call's
    // Stats slot is allocated but unused -- simpler and safer than special-casing the kernel.
    const size_t ragged_offsets_workspace_size = is_ragged ? 10 * num_bytes_per_ragged_offset : 0;
    if (workspace == nullptr) {
      *workspace_size =
          plan_workspace_size + actual_seqlen_workspace_size + ragged_offsets_workspace_size;
      return;
    }

    // cuDNN stream check needs to be moved here to support dummy kernel calls with
    // null streams for sizing the cuDNN workspace.
    NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

    // build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ},
        {K, devPtrK},
        {V, devPtrV},
        {O, devPtrO},
        {Stats, devPtrM},
        {dO, devPtrdO},
        {attn_scale, &scaling_factor},
        {descale_q, devPtrDescaleQ},
        {descale_k, devPtrDescaleK},
        {descale_v, devPtrDescaleV},
        {descale_dO, devPtrDescaledO},
        {dQ, devPtrdQ},
        {dK, devPtrdK},
        {dV, devPtrdV},
    };
    if (is_delayed_scaling || is_current_scaling) {
      variant_pack[descale_s] = devPtrDescaleS;
      variant_pack[descale_dP] = devPtrDescaledP;
      variant_pack[scale_s] = devPtrScaleS;
      variant_pack[scale_dP] = devPtrScaledP;
      variant_pack[amax_dP] = devPtrAmaxdP;
      variant_pack[amax_dQ] = devPtrAmaxdQ;
      variant_pack[amax_dK] = devPtrAmaxdK;
      variant_pack[amax_dV] = devPtrAmaxdV;
    }
    if (is_delayed_scaling || (is_current_scaling && !is_O_in_F16)) {
      variant_pack[descale_o] = devPtrDescaleO;
    }
    if (is_delayed_scaling) {
      variant_pack[scale_dQ] = devPtrScaledQ;
      variant_pack[scale_dK] = devPtrScaledK;
      variant_pack[scale_dV] = devPtrScaledV;
    }
    if (is_mxfp8) {
      variant_pack[Q_t] = devPtrQ_t;
      variant_pack[K_t] = devPtrK_t;
      variant_pack[dO_f16] = devPtrdO_f16;
      variant_pack[dO_t] = devPtrdO_t;
      variant_pack[descale_q_t] = devPtrDescaleQ_t;
      variant_pack[descale_k_t] = devPtrDescaleK_t;
      variant_pack[descale_dO_t] = devPtrDescaledO_t;
    }

    /* if (is_bias) {
       variant_pack[bias] = devPtrBias;
       if ((bias_b == 1) && (bias_h == h)) {
         variant_pack[dBias] = devPtrdBias;
       } else {
         variant_pack[dBias] = nullptr;
       }
    } */

    if (is_padding) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
      void* devActualSeqlenQ = static_cast<int8_t*>(workspace) + plan_workspace_size;
      void* devActualSeqlenKV = static_cast<int8_t*>(devActualSeqlenQ) + num_bytes_per_seqlen;
      // (actual_b, b): see fused_attn_fp8_fwd_impl.
      cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
          actual_b, b, static_cast<const int32_t*>(devPtrcuSeqlensQ),
          static_cast<const int32_t*>(devPtrcuSeqlensKV), static_cast<int32_t*>(devActualSeqlenQ),
          static_cast<int32_t*>(devActualSeqlenKV));
      NVTE_CHECK_CUDA(cudaGetLastError());
      variant_pack[seq_q] = devActualSeqlenQ;
      variant_pack[seq_kv] = devActualSeqlenKV;

      if (is_ragged) {
        // Element offsets into the packed buffers: offsets_x[i] = mult_x * cu_seqlens[i]. The
        // kernel derives the multipliers from layout_group/h/hg/d -- h*d for separate Q/K/V,
        // 3*h*d for t3hd/th3d, 2*hg*d for the KV-packed layouts, and h (not h*d) for Stats.
        //
        // b+1 entries, so its own launch extent; see fused_attn_fp8_fwd_impl.
        const size_t offsets_grid = (b + nthreads_per_block) / nthreads_per_block;
        int8_t* devOffsets =
            static_cast<int8_t*>(workspace) + plan_workspace_size + actual_seqlen_workspace_size;
        auto slot = [&](size_t i) {
          return static_cast<void*>(devOffsets + i * num_bytes_per_ragged_offset);
        };
        // Reads, from qkv_layout.
        void* devOffsetsQ = slot(0);
        void* devOffsetsK = slot(1);
        void* devOffsetsV = slot(2);
        void* devOffsetsO = slot(3);
        void* devOffsetsS = slot(4);
        // Gradients, from dqkv_layout. Slot 9 receives the second call's Stats output, which is
        // not consumed -- dO reuses the O slot of that set.
        void* devOffsetsdQ = slot(5);
        void* devOffsetsdK = slot(6);
        void* devOffsetsdV = slot(7);
        void* devOffsetsdO = slot(8);
        void* devOffsetsdS_unused = slot(9);

        // Ragged base offsets from the padded (physical) cumulative lengths, so each document is
        // addressed at its padded base; actual cu_seqlens above drives the per-document extents.
        cu_seqlens_padded_to_offsets<<<offsets_grid, nthreads_per_block, 0, stream>>>(
            nvte_get_qkv_layout_group(qkv_layout), actual_b, b, h, hg, d_qk, d_v,
            static_cast<const int32_t*>(devPtrcuSeqlensQPadded),
            static_cast<const int32_t*>(devPtrcuSeqlensKVPadded), ragged_offset_type, devOffsetsQ,
            devOffsetsK, devOffsetsV, devOffsetsO, devOffsetsS);
        NVTE_CHECK_CUDA(cudaGetLastError());

        // Second set, from dqkv_layout. Identical to the first when the layouts match; different
        // multipliers when they do not, which is the case this exists for.
        cu_seqlens_padded_to_offsets<<<offsets_grid, nthreads_per_block, 0, stream>>>(
            nvte_get_qkv_layout_group(dqkv_layout), actual_b, b, h, hg, d_qk, d_v,
            static_cast<const int32_t*>(devPtrcuSeqlensQPadded),
            static_cast<const int32_t*>(devPtrcuSeqlensKVPadded), ragged_offset_type, devOffsetsdQ,
            devOffsetsdK, devOffsetsdV, devOffsetsdO, devOffsetsdS_unused);
        NVTE_CHECK_CUDA(cudaGetLastError());

        variant_pack[offset_q] = devOffsetsQ;
        variant_pack[offset_k] = devOffsetsK;
        variant_pack[offset_v] = devOffsetsV;
        variant_pack[offset_o] = devOffsetsO;
        variant_pack[offset_stats] = devOffsetsS;
        variant_pack[offset_dq] = devOffsetsdQ;
        variant_pack[offset_dk] = devOffsetsdK;
        variant_pack[offset_dv] = devOffsetsdV;
        variant_pack[offset_do] = devOffsetsdO;
      }
    }

    if (is_dropout) {
      variant_pack[dropout_seed] = devPtrDropoutSeed;
      variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

    if (is_softmax_offset) {
      variant_pack[softmax_offset] = devPtrSoftmaxOffset;
      variant_pack[d_softmax_offset] = devPtrdSoftmaxOffset;
    }

    NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
  } catch (cudnn_frontend::cudnnException& e) {
    NVTE_ERROR(e.what());
  }
}  // NOLINT(readability/fn_size)

}  // namespace fused_attn

// fused attention FWD FP8 with separate Q, K, V
void fused_attn_fp8_fwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, bool is_training, float attn_scale,
    float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
    NVTE_QKV_Format qkv_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    NVTE_Softmax_Type softmax_type, size_t window_size_left, size_t window_size_right,
    bool bottom_right_diagonal, const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V,
    const Tensor* input_SoftmaxOffset, Tensor* input_output_S, Tensor* output_O,
    NVTETensorPack* Aux_CTX_Tensors, const Tensor* cu_seqlens_q, const Tensor* cu_seqlens_kv,
    const Tensor* cu_seqlens_q_padded, const Tensor* cu_seqlens_kv_padded, const Tensor* rng_state,
    Tensor* workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  // THD: Q is physically [t, h, d], so shape[0] is the total packed token count. Used below to
  // size the ragged Stats tensor; the graph's sequence buckets are computed inside the impl.
  const bool is_ragged_fmt = (nvte_get_qkv_format(qkv_layout) == NVTE_QKV_Format::NVTE_THD);
  const size_t num_tokens_q = is_ragged_fmt ? input_Q->data.shape[0] : 0;
  void *devPtrQ = nullptr, *devPtrK = nullptr, *devPtrV = nullptr;
  void *devPtrDescaleQ = nullptr, *devPtrDescaleK = nullptr, *devPtrDescaleV = nullptr;
  void *devPtrO = nullptr, *devPtrAmaxO = nullptr, *devPtrScaleO = nullptr;
  void *devPtrAmaxS = nullptr, *devPtrScaleS = nullptr, *devPtrDescaleS = nullptr;
  devPtrQ = input_Q->data.dptr;
  devPtrDescaleQ = input_Q->scale_inv.dptr;
  devPtrK = input_K->data.dptr;
  devPtrDescaleK = input_K->scale_inv.dptr;
  devPtrO = output_O->data.dptr;
  if (input_Q->scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
    devPtrV = input_V->data.dptr;
    devPtrDescaleV = input_V->scale_inv.dptr;
    devPtrScaleO = output_O->scale.dptr;
    devPtrAmaxS = input_output_S->amax.dptr;
    devPtrScaleS = input_output_S->scale.dptr;
    devPtrDescaleS = input_output_S->scale_inv.dptr;
    devPtrAmaxO = output_O->amax.dptr;
  } else if (input_Q->scaling_mode == NVTE_MXFP8_1D_SCALING) {
    devPtrV = input_V->columnwise_data.dptr;
    devPtrDescaleV = input_V->columnwise_scale_inv.dptr;
  }
  void* devPtrSoftmaxOffset = nullptr;
  if (softmax_type != NVTE_VANILLA_SOFTMAX) {
    devPtrSoftmaxOffset = input_SoftmaxOffset->data.dptr;
  }
  void* devPtrM = nullptr;
  if (Aux_CTX_Tensors->size == 0) {
    int i = 0;
    Tensor* output_M = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_M->data.dptr = nullptr;
    // Softmax stats are one value per (token, head). Under THD the caller allocates a packed
    // [t, h, 1] buffer -- the dense {b, h, s_q, 1} shape would be both the wrong size and the
    // wrong layout for a ragged graph, and the mismatch faults host-side in fused_attn_fwd.
    // Matches the F16 path (fused_attn_f16_arbitrary_seqlen.cu:1146).
    output_M->data.shape = is_ragged_fmt
                               ? std::vector<size_t>{num_tokens_q, num_attn_heads, 1}
                               : std::vector<size_t>{batch, num_attn_heads, max_seqlen_q, 1};
    output_M->data.dtype = DType::kFloat32;
    Tensor* output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_rng_state->data.dptr = nullptr;
    output_rng_state->data.shape = {2};
    output_rng_state->data.dtype = DType::kInt64;
    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      Tensor* output_softmax_offset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_softmax_offset->data.dptr = nullptr;
      output_softmax_offset->data.shape = {1, num_attn_heads, 1, 1};
      output_softmax_offset->data.dtype = DType::kFloat32;
    }
    Aux_CTX_Tensors->size = i;
  } else if (Aux_CTX_Tensors->size >= 2) {
    int i = 0;
    Tensor* output_M = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    devPtrM = output_M->data.dptr;
    Tensor* output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      Tensor* output_softmax_offset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_softmax_offset->data.dptr = devPtrSoftmaxOffset;
    }
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void* devPtrcuSeqlensQ =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_q->data.dptr));
  void* devPtrcuSeqlensKV =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_kv->data.dptr));
  // Ragged base offsets are built from the padded (physical) cumulative lengths; the actual
  // per-document extents (masking) come from cu_seqlens above. The dispatch always supplies a valid
  // padded tensor for the fused path -- equal to the actual cu_seqlens when the row is contiguous --
  // dereferenced directly, matching the F16 path (fused_attn_f16_arbitrary_seqlen.cu:1121).
  void* devPtrcuSeqlensQPadded =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_q_padded->data.dptr));
  void* devPtrcuSeqlensKVPadded =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_kv_padded->data.dptr));
  void* devPtrDropoutSeed =
      reinterpret_cast<void*>(reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset =
      reinterpret_cast<void*>(reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_Q->data.dtype;
  const DType O_type = output_O->data.dtype;
  size_t workspace_size = 0;

  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  // THD is admitted here as well: the impl now emits ragged offsets for Q/K/V/O/Stats. The
  // capability query (nvte_get_fused_attn_backend) is the gate that decides whether THD is
  // offered at all; this branch must agree with it or a selected backend becomes an error.
  if ((qkv_format == NVTE_QKV_Format::NVTE_BSHD) || (qkv_format == NVTE_QKV_Format::NVTE_SBHD) ||
      (qkv_format == NVTE_QKV_Format::NVTE_BHSD) || (qkv_format == NVTE_QKV_Format::NVTE_THD)) {
    fused_attn::fused_attn_fp8_fwd_impl(
        batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v,
        is_training, attn_scale, p_dropout, qkv_layout, o_format, bias_type, mask_type,
        softmax_type, window_size_left, window_size_right, bottom_right_diagonal, devPtrQ, devPtrK,
        devPtrV, devPtrSoftmaxOffset, devPtrM, devPtrO, devPtrDescaleQ, devPtrDescaleK,
        devPtrDescaleV, devPtrDescaleS, devPtrScaleS, devPtrScaleO, devPtrAmaxO, devPtrAmaxS,
        devPtrcuSeqlensQ, devPtrcuSeqlensKV, devPtrcuSeqlensQPadded, devPtrcuSeqlensKVPadded,
        devPtrDropoutSeed, devPtrDropoutOffset, get_cudnn_fe_dtype(QKV_type),
        get_cudnn_fe_dtype(O_type), input_Q->scaling_mode, qkv_scale_inv_format,
        workspace->data.dptr, &workspace_size, stream, handle);
  } else {
    NVTE_ERROR("FP8 fused attention only supports qkv_format=BSHD, SBHD, BHSD, or THD.\n");
  }

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  }
}
// fused attention BWD FP8 with separate Q, K, V
void fused_attn_fp8_bwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, float attn_scale, float p_dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format, NVTE_QKV_Format do_format,
    NVTE_QKV_Layout dqkv_layout, NVTE_QKV_Format qkv_scale_inv_format,
    NVTE_QKV_Format do_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    NVTE_Softmax_Type softmax_type, size_t window_size_left, size_t window_size_right,
    bool bottom_right_diagonal, bool deterministic, const Tensor* input_Q, const Tensor* input_K,
    const Tensor* input_V, const Tensor* input_O, const Tensor* input_dO,
    const Tensor* input_dO_f16, const Tensor* input_M, const Tensor* input_S,
    const Tensor* input_SoftmaxOffset, Tensor* input_output_dP, const Tensor* output_dQ,
    const Tensor* output_dK, const Tensor* output_dV, Tensor* output_dSoftmaxOffset,
    const Tensor* cu_seqlens_q, const Tensor* cu_seqlens_kv, const Tensor* cu_seqlens_q_padded,
    const Tensor* cu_seqlens_kv_padded, const Tensor* rng_state, Tensor* workspace,
    cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  void* devPtrQ = input_Q->data.dptr;
  void* devPtrK = input_K->data.dptr;
  void* devPtrV = input_V->data.dptr;
  void* devPtrDescaleQ = input_Q->scale_inv.dptr;
  void* devPtrDescaleK = input_K->scale_inv.dptr;
  void* devPtrDescaleV = input_V->scale_inv.dptr;
  void *devPtrQ_t = nullptr, *devPtrK_t = nullptr, *devPtrDescaleQ_t = nullptr,
       *devPtrDescaleK_t = nullptr;
  if (input_Q->scaling_mode == NVTE_MXFP8_1D_SCALING) {
    devPtrQ_t = input_Q->columnwise_data.dptr;
    devPtrDescaleQ_t = input_Q->columnwise_scale_inv.dptr;
    devPtrK_t = input_K->columnwise_data.dptr;
    devPtrDescaleK_t = input_K->columnwise_scale_inv.dptr;
  }

  void* devPtrO = input_O->data.dptr;
  const DType O_type = input_O->data.dtype;
  void* devPtrDescaleO = nullptr;
  if (O_type == DType::kFloat8E4M3 || O_type == DType::kFloat8E5M2) {
    devPtrDescaleO = input_O->scale_inv.dptr;
  }
  void* devPtrdO = input_dO->data.dptr;
  void* devPtrDescaledO = input_dO->scale_inv.dptr;
  void *devPtrdO_t = nullptr, *devPtrdO_f16 = nullptr, *devPtrDescaledO_t = nullptr;
  if (input_dO->scaling_mode == NVTE_MXFP8_1D_SCALING) {
    devPtrdO_t = input_dO->columnwise_data.dptr;
    devPtrdO_f16 = input_dO_f16->data.dptr;
    devPtrDescaledO_t = input_dO->columnwise_scale_inv.dptr;
  }

  void* devPtrM = input_M->data.dptr;

  void *devPtrScaleS = nullptr, *devPtrDescaleS = nullptr, *devPtrAmaxdP = nullptr,
       *devPtrScaledP = nullptr, *devPtrDescaledP = nullptr;
  if (input_Q->scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
    devPtrScaleS = input_S->scale.dptr;
    devPtrDescaleS = input_S->scale_inv.dptr;
    devPtrAmaxdP = input_output_dP->amax.dptr;
    devPtrScaledP = input_output_dP->scale.dptr;
    devPtrDescaledP = input_output_dP->scale_inv.dptr;
  }

  void* devPtrSoftmaxOffset = nullptr;
  void* devPtrdSoftmaxOffset = nullptr;
  if (softmax_type != NVTE_VANILLA_SOFTMAX) {
    devPtrSoftmaxOffset = input_SoftmaxOffset->data.dptr;
    devPtrdSoftmaxOffset = output_dSoftmaxOffset->data.dptr;
  }

  void* devPtrdQ = output_dQ->data.dptr;
  void* devPtrdK = output_dK->data.dptr;
  void* devPtrdV = output_dV->data.dptr;
  void *devPtrAmaxdQ = nullptr, *devPtrAmaxdK = nullptr, *devPtrAmaxdV = nullptr,
       *devPtrScaledQ = nullptr, *devPtrScaledK = nullptr, *devPtrScaledV = nullptr;
  if (input_Q->scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
    devPtrAmaxdQ = output_dQ->amax.dptr;
    devPtrAmaxdK = output_dK->amax.dptr;
    devPtrAmaxdV = output_dV->amax.dptr;
    devPtrScaledQ = output_dQ->scale.dptr;
    devPtrScaledK = output_dK->scale.dptr;
    devPtrScaledV = output_dV->scale.dptr;
  }

  void* devPtrcuSeqlensQ =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_q->data.dptr));
  void* devPtrcuSeqlensKV =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_kv->data.dptr));
  // Ragged base offsets from the padded cumulative lengths, dereferenced directly like the forward
  // and the F16 path; the dispatch always supplies a valid padded tensor (== actual when contiguous).
  void* devPtrcuSeqlensQPadded =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_q_padded->data.dptr));
  void* devPtrcuSeqlensKVPadded =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_kv_padded->data.dptr));
  void* devPtrDropoutSeed =
      reinterpret_cast<void*>(reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset =
      reinterpret_cast<void*>(reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_Q->data.dtype;
  const DType dO_type = input_dO->data.dtype;
  const DType dQKV_type = output_dQ->data.dtype;
  size_t workspace_size = 0;

  NVTE_QKV_Format dqkv_format = nvte_get_qkv_format(dqkv_layout);
  // THD admitted, mirroring the fprop guard at the top of fused_attn_fp8_fwd. As there, the
  // capability query (nvte_get_fused_attn_backend) is the real gate; this branch only has to
  // agree with it, or a backend the selector advertised turns into a hard error at execution.
  if ((dqkv_format == NVTE_QKV_Format::NVTE_BSHD) || (dqkv_format == NVTE_QKV_Format::NVTE_SBHD) ||
      (dqkv_format == NVTE_QKV_Format::NVTE_BHSD) || (dqkv_format == NVTE_QKV_Format::NVTE_THD)) {
    fused_attn::fused_attn_fp8_bwd_impl(
        batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v,
        attn_scale, p_dropout, qkv_layout, o_format, do_format, dqkv_layout, bias_type, mask_type,
        softmax_type, window_size_left, window_size_right, bottom_right_diagonal, deterministic,
        devPtrQ, devPtrK, devPtrV, devPtrM, devPtrO, devPtrdO, devPtrSoftmaxOffset, devPtrdQ,
        devPtrdK, devPtrdV, devPtrdSoftmaxOffset, devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
        devPtrDescaleO, devPtrDescaledO, devPtrDescaleS, devPtrDescaledP, devPtrScaleS,
        devPtrScaledP, devPtrScaledQ, devPtrScaledK, devPtrScaledV, devPtrAmaxdP, devPtrAmaxdQ,
        devPtrAmaxdK, devPtrAmaxdV, devPtrQ_t, devPtrK_t, devPtrdO_f16, devPtrdO_t,
        devPtrDescaleQ_t, devPtrDescaleK_t, devPtrDescaledO_t, devPtrcuSeqlensQ, devPtrcuSeqlensKV,
        devPtrcuSeqlensQPadded, devPtrcuSeqlensKVPadded, devPtrDropoutSeed, devPtrDropoutOffset,
        get_cudnn_fe_dtype(QKV_type), get_cudnn_fe_dtype(O_type), get_cudnn_fe_dtype(dO_type),
        get_cudnn_fe_dtype(dQKV_type), input_dO->scaling_mode, qkv_scale_inv_format,
        do_scale_inv_format, workspace->data.dptr, &workspace_size, stream, handle);
  } else {
    NVTE_ERROR("FP8 fused attention only supports dqkv_format=BSHD, SBHD, BHSD, or THD.\n");
  }

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  }
}
}  // namespace transformer_engine

NVTEFusedAttnFP8CacheStats nvte_get_fused_attn_fp8_cache_stats() {
  const auto& counters = transformer_engine::fused_attn::fp8_cache_stats;
  return {transformer_engine::fused_attn::fp8_cache_stats_enabled() ? 1 : 0,
          counters.fprop_lookups.load(std::memory_order_relaxed),
          counters.fprop_hits.load(std::memory_order_relaxed),
          counters.fprop_entries.load(std::memory_order_relaxed),
          counters.bprop_lookups.load(std::memory_order_relaxed),
          counters.bprop_hits.load(std::memory_order_relaxed),
          counters.bprop_entries.load(std::memory_order_relaxed)};
}

void nvte_reset_fused_attn_fp8_cache_stats() {
  auto& counters = transformer_engine::fused_attn::fp8_cache_stats;
  for (auto* counter : {&counters.fprop_lookups, &counters.fprop_hits, &counters.fprop_entries,
                        &counters.bprop_lookups, &counters.bprop_hits, &counters.bprop_entries}) {
    counter->store(0, std::memory_order_relaxed);
  }
}
