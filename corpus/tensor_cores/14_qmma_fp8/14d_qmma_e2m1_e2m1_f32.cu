// 14d_qmma_e2m1_e2m1_f32.cu
//
// Chapter 14 variant d. Same shape (m16n8k32) and accumulator (FP32) as 14a/b/c.
// Both inputs are E2M1 (FP4). Purpose: validate that QMMA opcode covers the
// extreme case of kind::f8f6f4 family (FP4), and identify control code bits
// for FP4 vs FP8 distinction.
//
// Matches CUTLASS SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>
// from mma_sm120.hpp (line 55 - the first struct in the file).
//
// Fragment layout per thread (identical register count to 14a/b/c, but
// different element packing):
//   A: uint32_t[4] = 4 * 8 e2m1 = 32 e2m1 per thread (vs 16 e4m3 in 14a)
//   B: uint32_t[2] = 2 * 8 e2m1 = 16 e2m1 per thread (vs 8 e4m3 in 14a)
//   C: float[4]
//   D: float[4]
//
// E2M1 1.0 = 0x2 (4 bits). Two nibbles per byte = 0x22. Packed 0x22222222.
//
// Compile: nvcc -arch=compute_120a -code=sm_120a -o 14d_qmma_e2m1_e2m1_f32 14d_qmma_e2m1_e2m1_f32.cu
// Dump:    cuobjdump --dump-sass 14d_qmma_e2m1_e2m1_f32 > 14d_qmma_e2m1_e2m1_f32.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void qmma_e2m1_e2m1_f32_kernel(
    const uint32_t* __restrict__ a,  // A: E2M1 (FP4)
    const uint32_t* __restrict__ b,  // B: E2M1 (FP4)
    const float*    __restrict__ c,
    float*          __restrict__ d)
{
    const unsigned tid = threadIdx.x;

    uint32_t a0 = a[tid * 4 + 0];
    uint32_t a1 = a[tid * 4 + 1];
    uint32_t a2 = a[tid * 4 + 2];
    uint32_t a3 = a[tid * 4 + 3];

    uint32_t b0 = b[tid * 2 + 0];
    uint32_t b1 = b[tid * 2 + 1];

    float c0 = c[tid * 4 + 0];
    float c1 = c[tid * 4 + 1];
    float c2 = c[tid * 4 + 2];
    float c3 = c[tid * 4 + 3];

    float d0, d1, d2, d3;

    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
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
    const int n_a = 128;
    const int n_b = 64;
    const int n_c = 128;
    const int n_d = 128;

    uint32_t *h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    float    *h_c = (float*)   malloc(n_c * sizeof(float));
    float    *h_d = (float*)   malloc(n_d * sizeof(float));

    // E2M1 1.0 = 0x2 (4 bits). Two nibbles per byte = 0x22. Packed 0x22222222.
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x22222222u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x22222222u;
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

    qmma_e2m1_e2m1_f32_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // Each D element = sum of 32 products of 1.0 * 1.0 = 32.0
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
