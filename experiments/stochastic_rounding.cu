#include <cuda_fp4.h>  // __nv_fp4_e2m1
#include <cstdint>

#include "cute/tensor.hpp"
#include "cutlass/numeric_conversion.h"

using namespace cute;

// CUTLASS_DEVICE
// cutlass::Array<cutlass::float_e2m1_t, 8> StochasticNumericConverterBase(
//     cutlass::Array<float, 8> const &input, cutlass::Array<uint32_t, 2> const &rbits) {
//     using result_type = cutlass::Array<cutlass::float_e2m1_t, 8>;
//     result_type output;
//     auto output_ptr = reinterpret_cast<uint16_t *>(&output);
//     asm volatile(
//         "{\n"
//         "cvt.rs.satfinite.e2m1x4.f32   %0, {%5, %4, %3, %2}, %10;\n"
//         "cvt.rs.satfinite.e2m1x4.f32   %1, {%9, %8, %7, %6}, %11;\n"
//         "}"
//         : "=h"(output_ptr[0]), "=h"(output_ptr[1])
//         : "f"(input[0]), "f"(input[1]), "f"(input[2]), "f"(input[3]), "f"(input[4]), "f"(input[5]),
//           "f"(input[6]), "f"(input[7]), "r"(rbits[0]), "r"(rbits[1]));
//     return output;
// }

// CUTLASS_DEVICE
// cutlass::Array<cutlass::float_e2m1_t, 16> StochasticNumericConverter(
//     cutlass::Array<float, 16> const &input, cutlass::Array<uint32_t, 4> const *rbits) {
//     using result_type = cutlass::Array<cutlass::float_e2m1_t, 16>;
//     result_type output;
//     cutlass::Array<cutlass::float_e2m1_t, 8> *result_ptr =
//         reinterpret_cast<cutlass::Array<cutlass::float_e2m1_t, 8> *>(&output);
//     cutlass::Array<float, 8> const *source_ptr =
//         reinterpret_cast<cutlass::Array<float, 8> const *>(&input);
//     cutlass::Array<uint32_t, 2> const *rbits_ptr =
//         reinterpret_cast<cutlass::Array<uint32_t, 2> const *>(rbits);
//     CUTLASS_PRAGMA_UNROLL
//     for (int i = 0; i < 2; i++) {
//         result_ptr[i] = StochasticNumericConverterBase(source_ptr[i], rbits_ptr[i]);
//     }
//     return output;
// }

static constexpr float FP4_VALS[16] = {
    0.0f,   // 0000
    0.5f,   // 0001
    1.0f,   // 0010
    1.5f,   // 0011
    2.0f,   // 0100
    3.0f,   // 0101
    4.0f,   // 0110
    6.0f,   // 0111
    -0.0f,  // 1000
    -0.5f,  // 1001
    -1.0f,  // 1010
    -1.5f,  // 1011
    -2.0f,  // 1100
    -3.0f,  // 1101
    -4.0f,  // 1110
    -6.0f   // 1111
};

int main() {
    using TC = cutlass::float_e2m1_t;
    using ElementAccumulator = float;

    constexpr int VectorSize = 4;
    float test_vals[4] = {0.0, .24, 1.25, 2.5};
    
    auto fp4_vals = __nv_fp4x4_e2m1(reinterpret_cast<float4*>(test_vals)[0]);
    __nv_fp4x4_storage_t &raw_fp4 = fp4_vals.__x;
    
    using FP32toFP4Converter =
        cutlass::NumericArrayConverter<TC, ElementAccumulator, VectorSize>;
    using SrcType = FP32toFP4Converter::source_type;
    using ResultType = FP32toFP4Converter::result_type;
    using CutlassFP4Type = ResultType::value_type;
    using Storage = CutlassFP4Type::Storage;

    auto numElements = ResultType::kStorageElements;
    auto x = ResultType::kElementsPerStoredItem;
    SrcType source;
    for(int i = 0; i < VectorSize; i++){
        source[i] = test_vals[i];
    }

    auto converter = FP32toFP4Converter{};
    auto result = converter.convert(source);
    
    for(int i = 0; i < 4; i++){
        uint8_t q = raw_fp4 >> 4*i & 0xF;
        float x = reinterpret_cast<float *>(test_vals)[i];
        uint8_t r = result[i].get();
        printf("%d: %f -> %x, %x => %f\n", i, test_vals[i], q, r, FP4_VALS[q]);
    }

    //   output_frgs[v] = {}(
    //       cutlass::multiplies<cutlass::Array<ElementAccumulator, VectorSize>>{}(compute_frgs[v],
    //                                                                             acc_scale));
}
