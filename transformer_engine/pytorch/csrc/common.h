/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/cast_transpose_noop.h>
#include <transformer_engine/comm_gemm_overlap.h>
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/fused_rope.h>
#include <transformer_engine/fused_router.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/multi_stream.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/normalization.h>
#include <transformer_engine/padding.h>
#include <transformer_engine/permutation.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/softmax.h>
#include <transformer_engine/swizzle.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/transpose.h>

#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <vector>

#include "c10/util/ArrayRef.h"
#include "common/util/logging.h"

namespace transformer_engine::pytorch {

// in python we have: dist_group_type = torch.distributed.ProcessGroup
using dist_group_type = c10d::ProcessGroup;

// Each tensor here is shape (N, ) holding all scaling
// data for a single FP8 block, e.g. LayerNormLinear
class FP8TensorMeta {
 public:
  at::Tensor scale;
  at::Tensor scale_inv;
  at::Tensor amax_history;
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8FwdTensors {
  GEMM1_INPUT = 0,
  GEMM1_WEIGHT = 1,
  GEMM1_OUTPUT = 2,
  GEMM2_INPUT = 3,
  GEMM2_WEIGHT = 4,
  GEMM2_OUTPUT = 5,
  GEMM3_INPUT = 6,
  GEMM3_WEIGHT = 7,
  GEMM3_OUTPUT = 8
};

// Used as named indices on the `scale`, `scale_inv`,
// and `amax` tensors in the `FP8TensorMeta` class.
enum FP8BwdTensors {
  GRAD_OUTPUT1 = 0,
  GRAD_INPUT1 = 1,
  GRAD_OUTPUT2 = 2,
  GRAD_INPUT2 = 3,
  GRAD_OUTPUT3 = 4,
  GRAD_INPUT3 = 5
};

class Quantizer {
 public:
  virtual NVTEScalingMode get_scaling_mode() const = 0;

  virtual void set_quantization_params(TensorWrapper* tensor) const = 0;

  /*! @brief Construct a tensor with uninitialized data */
  virtual std::pair<TensorWrapper, py::object> create_tensor(const std::vector<size_t>& shape,
                                                             DType dtype) const = 0;

  /*! @brief Convert a PyTorch tensor into a Transformer Engine C++ tensor
   *
   * The PyTorch tensor's attributes are modified to match the
   * quantizer's configuration.
   */
  virtual std::pair<TensorWrapper, py::object> convert_and_update_tensor(
      py::object tensor) const = 0;

  /*! @brief Convert to a quantized data format */
  virtual void quantize(const TensorWrapper& input, TensorWrapper& out,
                        const std::optional<TensorWrapper>& noop_flag = std::nullopt) = 0;

  virtual ~Quantizer() = default;

  bool rowwise_usage = true;
  bool columnwise_usage = true;
  bool internal = false;
  py::handle quantizer;

 protected:
  explicit Quantizer(const py::handle& quantizer);
};

class NoneQuantizer : public Quantizer {
 public:
  explicit NoneQuantizer(const py::handle& quantizer) : Quantizer(quantizer) {}

  NVTEScalingMode get_scaling_mode() const override { return NVTE_DELAYED_TENSOR_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override {}

  std::pair<TensorWrapper, py::object> create_tensor(const std::vector<size_t>& shape,
                                                     DType dtype) const override;

  /*! @brief Construct a tensor with pre-initialized data */
  std::pair<TensorWrapper, py::object> create_tensor(const std::vector<size_t>& shape, DType dtype,
                                                     at::Tensor data) const;

  std::pair<TensorWrapper, py::object> convert_and_update_tensor(py::object tensor) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;
};

class Float8Quantizer : public Quantizer {
 public:
  at::Tensor scale;
  at::Tensor scale_inv;
  at::Tensor amax;
  DType dtype;

  explicit Float8Quantizer(const py::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override { return NVTE_DELAYED_TENSOR_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override;

  std::pair<TensorWrapper, py::object> create_tensor(const std::vector<size_t>& shape,
                                                     DType dtype) const override;

  /*! @brief Construct a tensor with pre-initialized data */
  std::pair<TensorWrapper, py::object> create_tensor(const std::vector<size_t>& shape, DType dtype,
                                                     std::optional<at::Tensor> data,
                                                     std::optional<at::Tensor> transpose,
                                                     std::optional<at::Tensor> scale_inv) const;

  std::pair<TensorWrapper, py::object> convert_and_update_tensor(py::object shape) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;
};

class Float8CurrentScalingQuantizer : public Quantizer {
 public:
  at::Tensor scale;
  at::Tensor scale_inv;
  at::Tensor amax;
  DType dtype;
  bool with_amax_reduction;
  c10::intrusive_ptr<dist_group_type> amax_reduction_group;
  bool force_pow_2_scales = false;
  float amax_epsilon = 0.0;

  explicit Float8CurrentScalingQuantizer(const py::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override { return NVTE_DELAYED_TENSOR_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override;

  std::pair<TensorWrapper, py::object> create_tensor(const std::vector<size_t>& shape,
                                                     DType dtype) const override;

  /*! @brief Construct a high precision tensor giving it this quantizer's amax

  Note: this member function also zeros out the amax, as it is meant to be used in conjunction with
        a kernel computing the amax, which might expect the amax to be initialized to zero
  */
  std::pair<TensorWrapper, py::object> create_hp_tensor_with_amax(const std::vector<size_t>& shape,
                                                                  DType dtype);

  std::pair<TensorWrapper, py::object> convert_and_update_tensor(py::object shape) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;

  /*! @brief Convert to a quantized data format avoiding amax computation */
  void quantize_with_amax(TensorWrapper& input, TensorWrapper& out,
                          const std::optional<TensorWrapper>& noop_flag = std::nullopt);

 private:
  void quantize_impl(const TensorWrapper& input, TensorWrapper& out,
                     const std::optional<TensorWrapper>& noop_flag, bool compute_amax);
};

class Float8BlockQuantizer : public Quantizer {
 public:
  // Which float8 type is used for q data.
  DType dtype;
  // Options about how to quantize the tensor
  // Quantization scales are rounded down to powers of 2.
  bool force_pow_2_scales = false;
  // Amax within quantization tile has a floor of epsilon.
  float amax_epsilon = 0.0;
  // Whether quantized tensor will be used in an all-gather
  bool all_gather_usage = false;

 private:
  int block_scaling_dim = 2;

 public:
  // Initializes from a python handle to a Float8BlockQuantizer
  explicit Float8BlockQuantizer(const py::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override {
    return (block_scaling_dim == 2) ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D;
  }

  // Gets rowwise and columnwise_data from tensor and sets them on wrapper
  void set_quantization_params(TensorWrapper* tensor) const override;

  // Create a python Float8BlockQuantized tensor and C++ wrapper
  // for the tensor. Should set quantized data, scales for rowwise
  // and optionally columnwise usage.
  std::pair<TensorWrapper, py::object> create_tensor(const std::vector<size_t>& shape,
                                                     DType dtype) const override;

  std::pair<TensorWrapper, py::object> convert_and_update_tensor(py::object shape) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;

  std::vector<size_t> get_scale_shape(const std::vector<size_t>& shape, bool columnwise) const;
};

class MXFP8Quantizer : public Quantizer {
 public:
  DType dtype;

  explicit MXFP8Quantizer(const py::handle& quantizer);

  NVTEScalingMode get_scaling_mode() const override { return NVTE_MXFP8_1D_SCALING; }

  void set_quantization_params(TensorWrapper* tensor) const override;

  std::pair<TensorWrapper, py::object> create_tensor(const std::vector<size_t>& shape,
                                                     DType dtype) const override;

  std::pair<TensorWrapper, py::object> convert_and_update_tensor(py::object shape) const override;

  void quantize(const TensorWrapper& input, TensorWrapper& out,
                const std::optional<TensorWrapper>& noop_flag = std::nullopt) override;

  std::vector<size_t> get_scale_shape(const std::vector<size_t>& shape, bool columnwise) const;
};

std::unique_ptr<Quantizer> convert_quantizer(py::handle quantizer);

std::vector<size_t> getTensorShape(const at::Tensor& t);

transformer_engine::DType getTransformerEngineFP8Type(bool e4m3_if_hybrid,
                                                      const std::string& fp8_recipe);

inline size_t typeToNumBits(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt64:
      return 64;
    case transformer_engine::DType::kInt32:
    case transformer_engine::DType::kFloat32:
      return 32;
    case transformer_engine::DType::kInt16:
    case transformer_engine::DType::kFloat16:
    case transformer_engine::DType::kBFloat16:
      return 16;
    case transformer_engine::DType::kByte:
    case transformer_engine::DType::kFloat8E4M3:
    case transformer_engine::DType::kFloat8E5M2:
      return 8;
    case transformer_engine::DType::kFloat4E2M1:
      return 4;
    default:
      NVTE_ERROR("Invalid type");
  }
}

inline at::ScalarType GetATenDType(transformer_engine::DType t) {
  switch (t) {
    case transformer_engine::DType::kInt16:
      return torch::kInt16;
    case transformer_engine::DType::kInt32:
      return torch::kInt32;
    case transformer_engine::DType::kInt64:
      return torch::kInt64;
    case transformer_engine::DType::kFloat32:
      return at::kFloat;
    case transformer_engine::DType::kFloat16:
      return at::kHalf;
    case transformer_engine::DType::kBFloat16:
      return at::kBFloat16;
    case transformer_engine::DType::kByte:
      return at::kByte;
    case transformer_engine::DType::kFloat8E4M3:
      return at::kFloat8_e4m3fn;
    case transformer_engine::DType::kFloat8E5M2:
      return at::kFloat8_e5m2;
    default:
      NVTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(at::ScalarType t) {
  switch (t) {
    case at::kFloat8_e4m3fn:
      return transformer_engine::DType::kFloat8E4M3;
    case at::kFloat8_e5m2:
      return transformer_engine::DType::kFloat8E5M2;
    case at::kHalf:
      return transformer_engine::DType::kFloat16;
    case at::kFloat:
      return transformer_engine::DType::kFloat32;
    case at::kBFloat16:
      return transformer_engine::DType::kBFloat16;
    case at::kBool:
      return transformer_engine::DType::kByte;
    case torch::kByte:
      return transformer_engine::DType::kByte;
    case torch::kInt16:
      return transformer_engine::DType::kInt16;
    case torch::kInt32:
      return transformer_engine::DType::kInt32;
    case torch::kInt64:
      return transformer_engine::DType::kInt64;
    default:
      std::cout << "Type: " << static_cast<int>(t) << std::endl;
      NVTE_ERROR("Invalid type");
  }
}

inline transformer_engine::DType GetTransformerEngineDType(int DType_value) {
  return static_cast<transformer_engine::DType>(DType_value);
}

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const std::vector<size_t>& shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, const std::vector<size_t>& shape, const transformer_engine::DType type,
    void* amax_ptr, void* scale_ptr, void* scale_inv_ptr, std::vector<size_t> scale_inv_shape = {1},
    NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    void* data_ptr, void* columnwise_data_ptr, const std::vector<size_t>& shape,
    const std::vector<size_t>& columnwise_shape, const transformer_engine::DType type,
    void* amax_ptr, void* scale_ptr, void* scale_inv_ptr, void* columnwise_scale_inv_ptr,
    const std::vector<size_t>& scale_inv_shape = {1},
    const std::vector<size_t>& columnwise_scale_inv_shape = {1},
    NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

transformer_engine::TensorWrapper makeTransformerEngineTensor(void* data_ptr,
                                                              const NVTEShape& shape,
                                                              const transformer_engine::DType type);

transformer_engine::TensorWrapper makeTransformerEngineTensor(at::Tensor tensor);

std::tuple<std::vector<transformer_engine::TensorWrapper>, std::vector<std::vector<NVTETensor>>,
           std::vector<NVTETensor*>, size_t, size_t>
makeTransformerEngineTensorList(std::vector<std::vector<at::Tensor>> at_tensor_lists);

TensorWrapper makeTransformerEngineTensor(py::handle tensor, py::handle quantizer);

transformer_engine::TensorWrapper makeTransformerEngineTensor(
    at::Tensor tensor, at::Tensor amax, const at::Tensor scale, at::Tensor scale_inv,
    NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

template <typename T>
T product(const std::vector<T>& shape);

size_t product(const NVTEShape& shape, size_t begin, size_t end);

std::vector<size_t> nvte_shape_to_vector(const NVTEShape& nvte_shape);

at::Tensor allocateSpace(const std::vector<size_t>& shape, const transformer_engine::DType type,
                         bool init_to_zeros);

at::Tensor allocateSpace(const NVTEShape& shape, const transformer_engine::DType type,
                         bool init_to_zeros = false);

at::Tensor allocateTorchTensor(int M, int N, transformer_engine::DType dtype);

at::Tensor allocateTorchTensor(int M, transformer_engine::DType dtype);

void* getDataPtr(at::Tensor tensor, int offset = 0);

std::vector<size_t> convertShape(const NVTEShape& shape);

int roundup(const int value, const int multiple);

NVTEShape convertTorchShape(const c10::IntArrayRef torch_shape);
}  // namespace transformer_engine::pytorch

namespace std {
template <typename T>
string to_string(const vector<T>& vec) {
  string ret = "[";
  for (const auto& val : vec) {
    ret += to_string(val) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}

// Torch shape -> string
template <typename T>
string to_string(const c10::ArrayRef<T>& vec) {
  string ret = "[";
  for (const auto& val : vec) {
    ret += to_string(val) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}

inline string to_string(const NVTEShape& s) {
  string ret = "[";
  for (size_t i = 0; i < s.ndim; ++i) {
    ret += to_string(s.data[i]) + ",";
  }
  if (ret.size() > 1) {
    ret[ret.size() - 1] = ']';
  } else {
    ret += "]";
  }
  return ret;
}
}  // namespace std

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_COMMON_H_
