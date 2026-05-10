// 13b_hmma_f16_f32.cu
//
// Chapter 13 variant b. Same shape (m16n8k16), same FP16 inputs as 13a,
// but FP32 accumulator instead of FP16. Isolates the SASS delta induced
// by accumulator dtype alone.
//
// Matches CUTLASS SM80_16x8x16_F32F16F16F32_TN from mma_sm80.hpp (line 158).
//
// Fragment layout per thread (from CUTLASS):
//   A: uint32_t[4] = 8 half per thread   (unchanged from 13a)
//   B: uint32_t[2] = 4 half per thread   (unchanged from 13a)
//   C: float[4]    = 4 float per thread  (13a had uint32_t[2], now wider)
//   D: float[4]
//
// Warp-level: 32 threads cooperate on one 16x8 output tile (64 FP32 outputs).
//
// Compile: nvcc -arch=sm_120 -o 13b_hmma_f16_f32 13b_hmma_f16_f32.cu
// Dump:    cuobjdump --dump-sass 13b_hmma_f16_f32 > 13b_hmma_f16_f32.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void hmma_f16_f32_kernel(
    const uint32_t* __restrict__ a,  // 16x16 FP16 = 128 uint32 total
    const uint32_t* __restrict__ b,  // 16x8  FP16 =  64 uint32 total
    const float*    __restrict__ c,  // 16x8  FP32 = 128 float  total
    float*          __restrict__ d)  // 16x8  FP32 = 128 float  total
{
    const unsigned tid = threadIdx.x;

    // A: 4 uint32 per thread (same as 13a)
    uint32_t a0 = a[tid * 4 + 0];
    uint32_t a1 = a[tid * 4 + 1];
    uint32_t a2 = a[tid * 4 + 2];
    uint32_t a3 = a[tid * 4 + 3];

    // B: 2 uint32 per thread (same as 13a)
    uint32_t b0 = b[tid * 2 + 0];
    uint32_t b1 = b[tid * 2 + 1];

    // C: 4 float per thread (delta from 13a)
    float c0 = c[tid * 4 + 0];
    float c1 = c[tid * 4 + 1];
    float c2 = c[tid * 4 + 2];
    float c3 = c[tid * 4 + 3];

    // D accumulator output: 4 float per thread.
    float d0, d1, d2, d3;

    // The MMA under study. Exact form from CUTLASS SM80_16x8x16_F32F16F16F32_TN.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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
    const int n_a = 128;  // uint32 for A (FP16 packed)
    const int n_b = 64;   // uint32 for B (FP16 packed)
    const int n_c = 128;  // float  for C (FP32 scalar)
    const int n_d = 128;  // float  for D

    uint32_t *h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    float    *h_c = (float*)   malloc(n_c * sizeof(float));
    float    *h_d = (float*)   malloc(n_d * sizeof(float));

    // A and B: packed half = 1.0 (bit pattern 0x3c00, two per uint32 = 0x3c003c00)
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x3c003c00u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x3c003c00u;
    // C: zero initial accumulator
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

    hmma_f16_f32_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // Each D element = sum of 16 products of 1.0 * 1.0 = 16.0
    printf("d[0] = %f (expected 16.0)\n", h_d[0]);

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
