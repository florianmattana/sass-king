// 14a_qmma_e4m3_e4m3_f32.cu
//
// First QMMA chapter. Minimal MMA with FP8 E4M3 inputs and FP32 accumulator,
// shape m16n8k32. Heritage SM120 atom from kind::f8f6f4 family.
//
// Goal: observe the SASS opcode emitted for
//   mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
// and establish whether SM120 reuses HMMA or introduces a new opcode family.
//
// Matches CUTLASS SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>
// from include/cute/arch/mma_sm120.hpp.
//
// Fragment layout per thread (from CUTLASS):
//   A: uint32_t[4] = 16 e4m3 per thread (16x32 / 32 threads = 16 e4m3 / thread)
//   B: uint32_t[2] = 8 e4m3 per thread (32x8 / 32 threads = 8 e4m3 / thread)
//   C: float[4] = 4 float per thread (16x8 / 32 threads = 4 float / thread)
//   D: float[4]
//
// E4M3 bit pattern for 1.0 = 0x38 (per IEEE-like FP8 with 4-bit exp, 3-bit mantissa).
// Packed four per uint32 = 0x38383838.
//
// Compile: nvcc -arch=sm_120 -o 14a_qmma_e4m3_e4m3_f32 14a_qmma_e4m3_e4m3_f32.cu
// Dump:    cuobjdump --dump-sass 14a_qmma_e4m3_e4m3_f32 > 14a_qmma_e4m3_e4m3_f32.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void qmma_e4m3_e4m3_f32_kernel(
    const uint32_t* __restrict__ a,  // 16x32 E4M3 = 512 e4m3 = 128 uint32 total for the warp
    const uint32_t* __restrict__ b,  // 32x8  E4M3 = 256 e4m3 =  64 uint32 total
    const float*    __restrict__ c,  // 16x8  FP32 = 128 float
    float*          __restrict__ d)  // 16x8  FP32
{
    const unsigned tid = threadIdx.x;

    // A: 4 uint32 per thread (16 e4m3 per thread, since k=32)
    uint32_t a0 = a[tid * 4 + 0];
    uint32_t a1 = a[tid * 4 + 1];
    uint32_t a2 = a[tid * 4 + 2];
    uint32_t a3 = a[tid * 4 + 3];

    // B: 2 uint32 per thread (8 e4m3 per thread)
    uint32_t b0 = b[tid * 2 + 0];
    uint32_t b1 = b[tid * 2 + 1];

    // C: 4 float per thread (same as 13b)
    float c0 = c[tid * 4 + 0];
    float c1 = c[tid * 4 + 1];
    float c2 = c[tid * 4 + 2];
    float c3 = c[tid * 4 + 3];

    float d0, d1, d2, d3;

    // The MMA under study. Exact form from CUTLASS SM120_16x8x32_TN<e4m3, e4m3, f32>.
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1),
           "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );

    d[tid * 4 + 0] = d0;
    d[tid * 4 + 1] = d1;
    d[tid * 4 + 2] = d2;
    d[tid * 4 + 3] = d3;
}

int main(void) {
    const int n_a = 128;  // uint32 (16x32 e4m3 = 512 e4m3 = 128 uint32)
    const int n_b = 64;   // uint32 (32x8  e4m3 = 256 e4m3 =  64 uint32)
    const int n_c = 128;  // float
    const int n_d = 128;

    uint32_t *h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    float    *h_c = (float*)   malloc(n_c * sizeof(float));
    float    *h_d = (float*)   malloc(n_d * sizeof(float));

    // E4M3 1.0 = 0x38. Packed four per uint32 = 0x38383838.
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x38383838u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x38383838u;
    for (int i = 0; i < n_c; ++i) h_c[i] = 0.0f;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0.0f;

    uint32_t *d_a, *d_b;
    float    *d_c, *d_d;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t));
    cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_c, n_c * sizeof(float));
    cudaMalloc(&d_d, n_d * sizeof(float));

    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n_c * sizeof(float),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, n_d * sizeof(float),    cudaMemcpyHostToDevice);

    qmma_e4m3_e4m3_f32_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // Each D element = sum of 32 products of 1.0 * 1.0 = 32.0 (k=32 reduction)
    printf("d[0] = %f (expected 32.0)\n", h_d[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    return 0;
}
