import os
from collections import OrderedDict

import torch
import torch.nn as nn
import transformer_engine.pytorch as tex
from torch.nn import Parameter

from test_numerics import (
    model_configs,
)

from transformer_engine.pytorch import (
    GroupedLinear,
    Linear,
)
from transformer_engine.pytorch.cpp_extensions import general_gemm, general_grouped_gemm
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    fp8_autocast,
    fp8_model_init,
)
from transformer_engine.pytorch.module.base import get_multi_stream_cublas_workspace, get_workspace
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    is_bf16_compatible,
)

from viztracer import VizTracer
from types import ModuleType

from contextlib import nullcontext


class DummyTracer(nullcontext):
    def start(self):
        self.__enter__()
        return self

    def stop(self):
        self.__exit__()

    def save(self, *args, **kwargs):
        pass


batch_sizes = [1]

# Only run FP8 tests on supported devices.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)

sm_80plus = get_device_compute_capability() >= (8, 0)

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)

NVTE_TEST_NVINSPECT_ENABLED = int(os.environ.get("NVTE_TEST_NVINSPECT_ENABLED", "0"))
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# Record initial RNG state from script run.
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()

torch._dynamo.config.recompile_limit = 16


def get_module_root(module: str | ModuleType):
    from pathlib import Path
    import importlib

    if isinstance(module, str):
        try:
            module = importlib.import_module(module)
        except ImportError as e:
            raise ImportError(f"Could not import module: {module}") from e

    if not isinstance(module, ModuleType):
        raise TypeError("Input must be a module or a string representing a module.")

    if not hasattr(module, "__file__") or module.__file__ is None:
        raise AttributeError(
            f"The module '{module.__name__}' is a built-in module or a namespace package and has no file path."
        )

    return Path(module.__file__).parent.resolve().as_posix()


SHOULD_TRACE = os.getenv("RUN_TRACE", "0") == "1"
TRANSFORMER_ROOT = get_module_root("transformer_engine")


def setup_tracer(
    include_files=None,
    exclude_files=None,
    log_torch=True,
    ignore_c_function=False,
    ignore_frozen=True,
    log_func_args=True,
    log_func_retval=True,
    **kwargs,
):
    print(f"{include_files=} {exclude_files=}")
    tracer = VizTracer(
        include_files=include_files,
        exclude_files=exclude_files,
        log_torch=log_torch,
        ignore_c_function=ignore_c_function,
        ignore_frozen=ignore_frozen,
        log_func_args=log_func_args,
        log_func_retval=log_func_retval,
        **kwargs,
    )

    return tracer


def reset_rng_states() -> None:
    """revert back to initial RNG state."""
    torch.set_rng_state(_cpu_rng_state)
    torch.cuda.set_rng_state(_cuda_rng_state)


def get_split_sizes(num_gemms, seqlen, split_size=1):
    m = seqlen // split_size
    dist = torch.sort(torch.randint(0, m, (num_gemms - 2,))).values.tolist()
    dist.append(dist[-1])  # Manually add a zero
    m_splits = torch.tensor(dist + [m]) - torch.tensor([0] + dist)
    m_splits = m_splits * split_size
    assert m_splits.sum() == seqlen and len(m_splits) == num_gemms
    return m_splits


def _test_grouped_linear_accuracy(
    block,
    num_gemms,
    bs,
    dtype,
    config,
    recipe,
    fp8,
    fuse_wgrad_accumulation,
    delay_wgrad_compute=False,
    inp_hidden_states=None,
    m_splits=None,
):
    reset_rng_states()
    if fp8:
        FP8GlobalStateManager.reset()

    if inp_hidden_states is None:
        inp_hidden_states = torch.randn(
            (config.seq_len, bs, config.hidden_size),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        inp_hidden_states.retain_grad()

    if num_gemms > 1:
        if m_splits is None:
            split_size = 1
            if fp8:
                split_size = 16
                if recipe.mxfp8():
                    split_size = 128
            m = config.seq_len // split_size
            dist = torch.sort(torch.randint(0, m, (num_gemms - 2,))).values.tolist()
            dist.append(dist[-1])  # Manually add a zero
            m_splits = torch.tensor(dist + [m]) - torch.tensor([0] + dist)
            m_splits = m_splits * split_size
            assert m_splits.sum() == config.seq_len and len(m_splits) == num_gemms
    else:
        m_splits = torch.tensor([config.seq_len])

    with fp8_autocast(enabled=fp8, fp8_recipe=recipe):
        if isinstance(block, GroupedLinear):
            m_splits = m_splits * bs
            print(f"{m_splits=} {inp_hidden_states.shape=}")
            out = block.forward(inp_hidden_states, m_splits.tolist())
        else:
            out = torch.cat(
                [
                    block[i](inp)
                    for i, inp in enumerate(torch.split(inp_hidden_states, m_splits.tolist()))
                ]
            )

    loss = out.sum()
    loss.backward()

    if delay_wgrad_compute:
        if isinstance(block, GroupedLinear):
            block.backward_dw()
        else:
            for i in range(num_gemms):
                block[i].backward_dw()

    torch.cuda.synchronize()
    outputs = [out, inp_hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            if getattr(p, "main_grad", None) is not None:
                outputs.append(p.main_grad)
                assert p.grad is None  # grad should be None if fuse_wgrad_accumulation is True
            else:
                outputs.append(p.grad)
    return outputs


def run_sequential(sequential, inputs, m_splits):
    out = torch.cat(
        [sequential[i](inp) for i, inp in enumerate(torch.split(inputs, m_splits.tolist()))]
    )
    return out


# Notes:
# see /home/jeromeku/transformerengine/transformer_engine/pytorch/module/base.py reset_parameters
# also see float8_tensor and quantizedtensor subclasses for __torch_dispatch__ FSDP2 handling
# grouped_linear => general_grouped_gemm => multi_stream_cublas => for loop individual gemms on multiple streams
# weights are N x K k-major, inputs are M x K K major
# weights are operand A, inputs are operand B, layout is "TN" => (N x K) (K x M)
# "N" => for operand A => row-major is transposed so K = shape[0], M = shape[1]
#     => for operand B => N is shape[0], K = shape[1]
# Want M x N row-major out, equivalent to N x M col-major
# (M x N)_T = ((M x K) x (K x N))_T => (N x K) x (K x M)
def test_grouped_linear_accuracy(
    dtype=torch.float32,
    num_gemms=4,
    bs=1,
    model="126m",
    recipe=None,
    fp8_model_params=False,
    fuse_wgrad_accumulation=False,
    bias=False,
    delay_wgrad_compute=False,
    parallel_mode=None,
):
    fp8 = recipe is not None

    config = model_configs[model]

    with fp8_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
        grouped_linear = GroupedLinear(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            bias=bias,
            params_dtype=dtype,
            parallel_mode=parallel_mode,
            device="cuda",
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            delay_wgrad_compute=delay_wgrad_compute,
        ).eval()

        sequential_linear = torch.nn.ModuleList(
            [
                Linear(
                    config.hidden_size,
                    4 * config.hidden_size,
                    bias=bias,
                    params_dtype=dtype,
                    parallel_mode=parallel_mode,
                    device="cuda",
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                ).eval()
                for _ in range(num_gemms)
            ]
        )

    # Share params
    with torch.no_grad():
        for i in range(num_gemms):
            sequential_linear[i].weight = Parameter(getattr(grouped_linear, f"weight{i}").clone())
            if bias:
                sequential_linear[i].bias = Parameter(getattr(grouped_linear, f"bias{i}").clone())
            if fuse_wgrad_accumulation:
                weight_i = getattr(grouped_linear, f"weight{i}")
                weight_i.main_grad = torch.rand_like(weight_i, dtype=torch.float32)
                sequential_linear[i].weight.main_grad = weight_i.main_grad.clone()

    inp_hidden_states = torch.randn(
        (config.seq_len, bs, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    m_sizes = get_split_sizes(num_gemms, config.seq_len, split_size=1)
    print(f"{m_sizes=}")
    m_splits = m_sizes
    m_sizes = m_sizes.tolist()

    outputs_ref = _test_grouped_linear_accuracy(
        sequential_linear,
        num_gemms,
        bs,
        dtype,
        config,
        recipe,
        fp8,
        fuse_wgrad_accumulation,
        delay_wgrad_compute,
        inp_hidden_states=inp_hidden_states,
        m_splits=m_splits,
    )
    outputs = _test_grouped_linear_accuracy(
        grouped_linear,
        num_gemms,
        bs,
        dtype,
        config,
        recipe,
        fp8,
        fuse_wgrad_accumulation,
        delay_wgrad_compute,
        inp_hidden_states=inp_hidden_states,
        m_splits=m_splits,
    )
    out = grouped_linear(inp_hidden_states, m_sizes)
    seq_out = run_sequential(sequential_linear, inp_hidden_states, m_splits)

    grouped_ref_out = outputs[0]
    seq_ref_out = outputs_ref[0]

    diff = (out - grouped_ref_out).abs().max()
    print(f"grouped diff: {diff.item():.4f}")
    diff = (out - seq_ref_out).abs().max()
    print(f"grouped vs seq diff: {diff.item():.4f}")
    diff = (seq_out - seq_ref_out).abs().max()
    print(f"seq ref vs seq diff: {diff.item():.4f}")

    # Should be bit-wise match
    for i, (o, o_ref) in enumerate(zip(outputs, outputs_ref)):
        torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


def bench_grouped_linear(
    dtype=torch.float32,
    num_gemms=4,
    bias=False,
    bs=1,
    seqlen=1024,
    hidden_size=768,
    parallel_mode=None,
):
    from triton.testing import do_bench

    hidden_size = 768
    grouped_linear = GroupedLinear(
        num_gemms,
        hidden_size,
        4 * hidden_size,
        bias=bias,
        params_dtype=dtype,
        parallel_mode=parallel_mode,
        device="cuda",
        fuse_wgrad_accumulation=False,
        delay_wgrad_compute=False,
    ).eval()

    sequential_linear = torch.nn.ModuleList(
        [
            Linear(
                hidden_size,
                4 * hidden_size,
                bias=bias,
                params_dtype=dtype,
                parallel_mode=parallel_mode,
                device="cuda",
                fuse_wgrad_accumulation=False,
            ).eval()
            for _ in range(num_gemms)
        ]
    )

    # Share params
    with torch.no_grad():
        for i in range(num_gemms):
            sequential_linear[i].weight = Parameter(getattr(grouped_linear, f"weight{i}").clone())
            if bias:
                sequential_linear[i].bias = Parameter(getattr(grouped_linear, f"bias{i}").clone())

    inp_hidden_states = torch.randn(
        (seqlen, bs, hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    inp_hidden_states.retain_grad()

    m_sizes = get_split_sizes(num_gemms, seqlen, split_size=1)
    print(f"{m_sizes=}")
    m_splits = m_sizes
    m_sizes = m_sizes.tolist()

    out = grouped_linear(inp_hidden_states, m_sizes)
    seq_out = run_sequential(sequential_linear, inp_hidden_states, m_splits)
    assert out.allclose(seq_out)
    
    def run_grouped_linear(grouped_linear, inputs: torch.Tensor, m_splits: torch.Tensor):
        return grouped_linear(inputs, m_splits.tolist())
    
    g_time = do_bench(lambda: run_grouped_linear(grouped_linear, inp_hidden_states, m_splits))
    s_time = do_bench(lambda: run_sequential(sequential_linear, inp_hidden_states, m_splits))
    print(f"{g_time:.4f}ms | {s_time:.4f}ms: {s_time / g_time:.1f}x")

if __name__ == "__main__":
    dtype = torch.float32
    num_gemms = 4
    bs = 1
    seqlen = 1024
    hidden_size = 768
    # test_grouped_linear_accuracy(dtype=dtype, num_gemms=num_gemms, bs=batch_size)
    bench_grouped_linear(dtype=dtype,num_gemms=num_gemms, bs=bs, seqlen=seqlen, hidden_size=hidden_size)

# @pytest.mark.parametrize("recipe", fp8_recipes + [None])
# def test_grouped_linear_accuracy_single_gemm(recipe):
#     """Split the tests to save CI time"""
#     test_grouped_linear_accuracy(
#         dtype=torch.float32,
#         num_gemms=1,
#         bs=2,
#         model="126m",
#         recipe=recipe,
#         fp8_model_params=True,
#         fuse_wgrad_accumulation=True,
#         bias=True,
#         delay_wgrad_compute=False,
#     )


# def _test_padding_grouped_linear_accuracy(block, num_gemms, bs, dtype, config, recipe, fp8=False):
#     def _pad_tensor_for_fp8(hidden_states, tokens_per_expert):
#         align_size = 16
#         if recipe.mxfp8():
#             align_size = 32
#         padded_tokens_per_expert = [
#             (num_tokens + align_size - 1) // align_size * align_size
#             for num_tokens in tokens_per_expert
#         ]
#         hidden_states = torch.split(hidden_states, tokens_per_expert)
#         padded_hidden_states = []
#         for hidden_state, actual_num_tokens, padded_num_tokens in zip(
#             hidden_states, tokens_per_expert, padded_tokens_per_expert
#         ):
#             padded_hidden_states.append(hidden_state)
#             if padded_num_tokens > actual_num_tokens:
#                 pad_tensor = torch.zeros(
#                     padded_num_tokens - actual_num_tokens,
#                     hidden_state.shape[1],
#                     dtype=hidden_state.dtype,
#                     device=hidden_state.device,
#                 )
#                 padded_hidden_states.append(pad_tensor)
#         padded_hidden_states = torch.cat(padded_hidden_states, dim=0)
#         return padded_hidden_states, padded_tokens_per_expert

#     def _unpad_tensor_for_fp8(padded_hidden_states, actual_tokens_per_expert, tokens_per_expert):
#         inputmats = torch.split(
#             padded_hidden_states.view(-1, padded_hidden_states.shape[-1]), tokens_per_expert
#         )
#         hidden_states = torch.cat(
#             [
#                 grad_output_mat[: actual_tokens_per_expert[i]]
#                 for i, grad_output_mat in enumerate(inputmats)
#             ],
#             dim=0,
#         )

#         return hidden_states

#     def _generate_random_numbers(n, total_sum):
#         if n <= 0:
#             return []

#         # reset seed
#         random.seed(seed)

#         breaks = sorted(random.sample(range(1, total_sum), n - 1))
#         random_numbers = (
#             [breaks[0]]
#             + [breaks[i] - breaks[i - 1] for i in range(1, n - 1)]
#             + [total_sum - breaks[-1]]
#         )

#         return random_numbers

#     reset_rng_states()
#     if fp8:
#         FP8GlobalStateManager.reset()

#     inp_hidden_states = torch.randn(
#         (config.seq_len * bs, config.hidden_size),
#         dtype=dtype,
#         device="cuda",
#         requires_grad=True,
#     )
#     inp_hidden_states.retain_grad()

#     m_splits = _generate_random_numbers(num_gemms, config.seq_len * bs)

#     with fp8_autocast(enabled=fp8, fp8_recipe=recipe):
#         if isinstance(block, TorchGroupedLinearWithPadding):
#             out = block(inp_hidden_states, m_splits)
#         else:
#             if fp8:
#                 padded_inp_hidden_states, padding_m_splits = _pad_tensor_for_fp8(
#                     inp_hidden_states, m_splits
#                 )
#                 padded_inp_hidden_states = block(padded_inp_hidden_states, padding_m_splits)
#                 out = _unpad_tensor_for_fp8(padded_inp_hidden_states, m_splits, padding_m_splits)
#             else:
#                 out = block(inp_hidden_states, m_splits)

#     loss = out.sum()
#     loss.backward()

#     torch.cuda.synchronize()
#     outputs = [out, inp_hidden_states.grad]
#     for p in block.parameters():
#         if p.requires_grad:
#             outputs.append(p.grad)
#     return outputs


# @pytest.mark.parametrize("dtype", param_types)
# @pytest.mark.parametrize("num_gemms", [3, 6])
# @pytest.mark.parametrize("bs", [1])
# @pytest.mark.parametrize("model", ["126m"])
# @pytest.mark.parametrize("fp8", [True])
# @pytest.mark.parametrize("recipe", fp8_recipes)
# @pytest.mark.parametrize("fp8_model_params", all_boolean)
# def test_padding_grouped_linear_accuracy(
#     dtype, num_gemms, bs, model, fp8, recipe, fp8_model_params, parallel_mode=None
# ):
#     if fp8 and not fp8_available:
#         pytest.skip(reason_for_no_fp8)
#     if recipe.mxfp8() and not mxfp8_available:
#         pytest.skip(reason_for_no_mxfp8)
#     if fp8_model_params and NVTE_TEST_NVINSPECT_ENABLED:
#         pytest.skip("FP8 parameters are not supported in debug mode.")
#     if recipe.float8_block_scaling() and not fp8_block_scaling_available:
#         pytest.skip(reason_for_no_fp8_block_scaling)

#     config = model_configs[model]
#     if config.seq_len % 16 != 0 and fp8:
#         pytest.skip("FP8 requires sequence length to be divisible by 16.")

#     with fp8_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
#         grouped_linear = TorchGroupedLinearWithPadding(
#             num_gemms,
#             config.hidden_size,
#             4 * config.hidden_size,
#             bias=False,
#             params_dtype=dtype,
#             parallel_mode=parallel_mode,
#             fp8=fp8,
#         ).eval()

#     with fp8_model_init(enabled=fp8 and fp8_model_params, recipe=recipe):
#         ref_grouped_linear = GroupedLinear(
#             num_gemms,
#             config.hidden_size,
#             4 * config.hidden_size,
#             bias=False,
#             params_dtype=dtype,
#             parallel_mode=parallel_mode,
#             device="cuda",
#         ).eval()

#     # Share params
#     with torch.no_grad():
#         inner_grouped_linear = grouped_linear.linear_fn
#         for i in range(num_gemms):
#             setattr(
#                 ref_grouped_linear,
#                 f"weight{i}",
#                 Parameter(getattr(inner_grouped_linear, f"weight{i}").clone()),
#             )

#     outputs = _test_padding_grouped_linear_accuracy(
#         grouped_linear, num_gemms, bs, dtype, config, recipe, fp8
#     )
#     outputs_ref = _test_padding_grouped_linear_accuracy(
#         ref_grouped_linear, num_gemms, bs, dtype, config, recipe, fp8
#     )

#     # Shoule be bit-wise match
#     for i, (o, o_ref) in enumerate(zip(outputs, outputs_ref)):
#         torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


# @pytest.mark.parametrize(
#     "shape",
#     [
#         (1, 127, 128, 512),
#         (8, 15, 128, 512),
#         (8, 1027, 128, 512),
#         (16, 10027, 128, 512),
#     ],
# )
# @pytest.mark.parametrize("dtype", param_types)
# @pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
# @pytest.mark.parametrize("accumulate", [False, True])
# def test_grouped_gemm(shape, dtype, layout, accumulate):
#     torch.manual_seed(0)
#     z, m, k, n = shape

#     dist = torch.sort(torch.randint(0, m, (z - 1,))).values.tolist()
#     m_splits = torch.tensor(dist + [m]) - torch.tensor([0] + dist)
#     assert m_splits.sum() == m and len(m_splits) == z
#     m_splits = m_splits.tolist()

#     if layout == "TN":
#         A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
#         B = list(torch.split(torch.randn(m, k, dtype=dtype, device="cuda"), m_splits))  # input
#         out = [torch.randn(m, n, dtype=dtype, device="cuda")]  # output
#         out_ref = [o.clone() for o in torch.split(out[0], m_splits)]
#         grad = False
#         single_output = True
#     elif layout == "NN":
#         A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
#         B = list(
#             torch.split(torch.randn(m, n, dtype=dtype, device="cuda"), m_splits)
#         )  # grad_output
#         out = [torch.randn(m, k, dtype=dtype, device="cuda")]  # dgrad
#         out_ref = [o.clone() for o in torch.split(out[0], m_splits)]
#         grad = True
#         single_output = True
#     else:  # layout == "NT"
#         A = list(torch.split(torch.randn(m, k, dtype=dtype, device="cuda"), m_splits))  # input
#         B = list(
#             torch.split(torch.randn(m, n, dtype=dtype, device="cuda"), m_splits)
#         )  # grad_output
#         out = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # wgrad
#         out_ref = [o.clone() for o in out]
#         grad = True
#         single_output = False

#     for i in range(z):
#         general_gemm(
#             A[i],
#             B[i],
#             get_workspace(),
#             dtype,
#             grad=grad,
#             accumulate=accumulate,
#             layout=layout,
#             out=out_ref[i],
#         )
#     if single_output:
#         out_ref = [torch.cat(out_ref)]

#     general_grouped_gemm(
#         A,
#         B,
#         out,
#         dtype,
#         get_multi_stream_cublas_workspace(),
#         m_splits=m_splits,
#         grad=grad,
#         accumulate=accumulate,
#         layout=layout,
#         single_output=single_output,
#     )

#     # should be bit-wise match
#     for o, o_ref in zip(out, out_ref):
#         torch.testing.assert_close(o, o_ref, rtol=0, atol=0)


# @pytest.mark.parametrize(
#     "shape",
#     [
#         (1, 128, 128, 512),
#         (8, 1024, 128, 512),
#         (16, 4096, 128, 512),
#     ],
# )
# @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2])
# @pytest.mark.parametrize("accumulate", [False, True])
# def test_fp8_grouped_gemm(shape, fp8_dtype, accumulate):
#     if not fp8_available:
#         pytest.skip(reason_for_no_fp8)

#     z, m, k, n = shape
#     m_splits = [m // z] * z

#     dtype = torch.bfloat16
#     A = [torch.randn(n, k, dtype=dtype, device="cuda") for _ in range(z)]  # weight
#     B = torch.split(torch.randn(m, k, dtype=dtype, device="cuda"), m_splits)  # input
#     out = torch.split(torch.randn(m, n, dtype=dtype, device="cuda"), m_splits)  # output
#     out_ref = [o.clone() for o in out]

#     # fp8 should be robust enough to this fake scale
#     scale = 1 + torch.rand(1, dtype=torch.float32, device="cuda").squeeze()
#     amax = torch.zeros(1, 1, dtype=torch.float32, device="cuda")

#     a_quantizers = [
#         Float8Quantizer(
#             scale.clone(),
#             amax.clone(),
#             tex.DType.kFloat8E4M3,
#         )
#         for _ in range(z)
#     ]
#     b_quantizers = [
#         Float8Quantizer(
#             scale.clone(),
#             amax.clone(),
#             tex.DType.kFloat8E4M3,
#         )
#         for _ in range(z)
#     ]

#     A_fp8 = []
#     B_fp8 = []

#     for i in range(z):
#         A_fp8.append(a_quantizers[i](A[i]))
#         B_fp8.append(b_quantizers[i](B[i]))

#     # baseline
#     for i in range(z):
#         general_gemm(
#             A_fp8[i],
#             B_fp8[i],
#             get_workspace(),
#             dtype,
#             out=out_ref[i],
#             accumulate=accumulate,
#         )
#     general_grouped_gemm(
#         A_fp8,
#         B_fp8,
#         out,
#         dtype,
#         get_multi_stream_cublas_workspace(),
#         m_splits=m_splits,
#         accumulate=accumulate,
#     )

#     # should be bit-wise match
#     for o, o_ref in zip(out, out_ref):
#         torch.testing.assert_close(o, o_ref, rtol=0, atol=0)
