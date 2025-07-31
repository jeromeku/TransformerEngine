// nvcc -std=c++17 -arch=sm_80 -O3 test_bf16.cu -o test_bf16
// run: ./test_bf16
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>   // __half / FP16 intrinsics

template <typename T>             // generic version: keep FP32
__host__ __device__ inline T cast_from_float(float v) { return v; }

template <>                       // FP16
__host__ __device__ inline __half cast_from_float<__half>(float v) {
    return __float2half_rn(v);    // round-to-nearest-even
}

template <>                       // BF16
__host__ __device__ inline __nv_bfloat16 cast_from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);   // RTNE  :contentReference[oaicite:0]{index=0}
}

template <typename T>             // generic version: keep FP32
__host__ __device__ inline float cast_to_float(T v) { return v; }

template <>                       // BF16
__host__ __device__ inline float cast_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);   // RTNE  :contentReference[oaicite:0]{index=0}
}

template <>                       // FP16
__host__ __device__ inline float cast_to_float<__half>(__half v) {
    return __half2float(v);    // round-to-nearest-even
}


template<int N, typename weight_t>
__global__ void rms_cast_kernel(const float*  __restrict__ x,
                                const float*  __restrict__ gamma,
                                weight_t*       z,
                                float rs,
                                bool zero_centered)
{
    int i = threadIdx.x;
    if (i >= N) return;

    float  y = rs * x[i];
    float  g = zero_centered ? (gamma[i] + 1.0f) : gamma[i];

    weight_t y_w = cast_from_float<weight_t>(y);
    weight_t g_w = cast_from_float<weight_t>(g);
    z[i] = g_w * y_w;                          
}

int main() {
    constexpr int N = 4;
    float  h_x[N]     = {1.f, 2.f, 3.f, 4.f};
    float  h_gamma[N] = {0.1f, 0.2f, 0.3f, 0.4f};
    float  rs = 2.f;
    bool   zero_centered = false;
    using weight_t = __nv_bfloat16;

    // device buffers
    float *d_x, *d_g;
    weight_t *d_z;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_g, N*sizeof(float));
    cudaMalloc(&d_z, N*sizeof(weight_t));
    cudaMemcpy(d_x, h_x,     sizeof(h_x),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_gamma, sizeof(h_gamma), cudaMemcpyHostToDevice);

    rms_cast_kernel<N><<<1,N>>>(d_x, d_g, d_z, rs, zero_centered);
    cudaDeviceSynchronize();

    // fetch & validate
    weight_t h_z[N];
    cudaMemcpy(h_z, d_z, sizeof(h_z), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float zf = cast_to_float<weight_t>(h_z[i]);        // BF16 → FP32 for print
        float ref = h_gamma[i] * rs * h_x[i];
        printf("idx %d : GPU %.6f  |  Ref %.6f\n", i, zf, ref);
        ok &= fabsf(zf - ref) < 1e-2f;              // BF16 ≈ 1 ulp ≈ 1e-2
    }
    printf(ok ? "✓ all good\n" : "✗ mismatch\n");

    cudaFree(d_x); cudaFree(d_g); cudaFree(d_z);
    return ok ? 0 : 1;
}
