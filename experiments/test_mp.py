import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling
torch.autograd.set_multithreading_enabled(False)

torch.cuda.set_device(0)
torch.manual_seed(12345)

fp8_recipe = NVFP4BlockScaling()
# NOTE: default recipe applies RHT to input / output fwd gemm which requires weights / activations in bfloat16 (see test_nvfp4_module_exact test)
high_precision_dtype = torch.bfloat16
my_linear = te.Linear(768, 2048, params_dtype=high_precision_dtype, bias=False)

inp = torch.rand((1024, 768), dtype=high_precision_dtype, requires_grad=True).cuda()
with te.autocast(enabled=True, recipe=fp8_recipe):
    out_fp8 = my_linear(inp)
loss = out_fp8.mean()
#breakpoint()
loss.backward()

"""    
/home/jk/transformerengine/transformer_engine/pytorch/tensor/nvfp4_tensor.py:193-194
nvfp4 scale factor layout and swizzling
b /home/jk/transformerengine/transformer_engine/pytorch/csrc/extensions/cast.cpp:36
b NVFP4Quantizer::quantize
b NVFP4Quantizer::quantize_with_amax
b nvte_quantize_v2
b NVFP4Quantizer::create_tensor
b NVFP4Quantizer::convert_and_update_tensor
b NVFP4Quantizer::NVFP4Quantizer
b quantize_fwd_helper
b quantize_transpose_vector_blockwise_fp4
b nvfp4::quantize_transpose

b -n NVFP4Quantizer::quantize -n NVFP4Quantizer::quantize_with_amax
  -n nvte_quantize_v2 \
  -n NVFP4Quantizer::create_tensor \
  -n NVFP4Quantizer::convert_and_update_tensor \
  -n NVFP4Quantizer::NVFP4Quantizer \
  -n quantize_fwd_helper \
  -n quantize_transpose_vector_blockwise_fp4 \
  -n nvfp4::quantize_transpose
b -n NVFP4Quantizer::quantize -n NVFP4Quantizer::quantize_with_amax -n nvte_quantize_v2 -n NVFP4Quantizer::create_tensor -n NVFP4Quantizer::convert_and_update_tensor -n NVFP4Quantizer::NVFP4Quantizer -n quantize_fwd_helper -n quantize_transpose_vector_blockwise_fp4 -n nvfp4::quantize_transpose -n hadamard_transform_amax
  
  br set -r '^(NVFP4Quantizer::quantize(_with_amax)?|nvte_quantize_v2|NVFP4Quantizer::create_tensor|NVFP4Quantizer::convert_and_update_tensor|NVFP4Quantizer::NVFP4Quantizer|quantize_fwd_helper|quantize_transpose_vector_blockwise_fp4|nvfp4::quantize_transpose)$'

get_scale_shape  
/home/jk/transformerengine/transformer_engine/pytorch/csrc/quantizer.cpp:1699-1709

with_post_rht_amax:
/home/jk/transformerengine/transformer_engine/pytorch/csrc/quantizer.cpp:1493-1500

hadamard kernel:
template <typename IType, int kHadamardDimension, int CHUNK_DIM_Y, int CHUNK_DIM_X, int BUFF_DIM_Y,
          int BUFF_DIM_X, int THREADS_PER_CHUNK, int THREADS_PER_Y, bool kReturnPreRhtAmax,
          bool kReturnIdentityAmax, bool kReturnTransposedAmax>

IType: bfloat16
chunk = 128
buff = 64
kThreadBlockX = 4
ThreadsPerWarp = 32
kThreadBlockY = 1
THREADS_PER_CHUNK = kThreadBlockX * kThreadsPerWarp = 128

block(4 * 32, 1)

ReturnPreRhtAmax = True, ReturnId = False, ReturnTransposedAmax = True

auto kernel = HadamardAmaxTmaKernel<
    IType, kHadamardDimension, kChunkBlockYSmall, kChunkBlockXSmall, kBuffDimY,
    kBuffDimX, kThreadBlockX * kThreadsPerWarp, kThreadBlockY, kReturnPreRhtAmax,
    kReturnIdentityAmax, kReturnTransposedAmax>;          
"""