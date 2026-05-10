// 14g_qmma_e3m2_e3m2_f32.cu
//
// Chapter 14 variant g. QMMA with FP6 inputs (e3m2). Same shape (m16n8k32) and
// accumulator (FP32) as 14a/b/c/d. Purpose: identify control code bits that
// distinguish FP6 family from FP8 (E4M3/E5M2) and FP4 (E2M1) within kind::f8f6f4.
//
// Matches CUTLASS SM120_16x8x32_TN<float_e3m2_t, float_e3m2_t, float>
// from mma_sm120.hpp (line 258).
//
// Fragment layout per thread (identical register count to 14a/b/c/d):
//   A: uint32_t[4] = 16 e3m2 per thread, but FP6 packing is 6 bits per element
//      so the actual mapping of bits to elements is non-trivial.
//   B: uint32_t[2]
//   C: float[4]
//   D: float[4]
//
// Note on correctness: the FP6 packing into uint32 follows a CUTLASS-internal
// scheme that is not trivially "5.33 elements per uint32". We use 0x00000000
// inputs to get a deterministic (zero) accumulation, sidestepping the packing
// question. The SASS observation (opcode, control code) does not depend on
// input values.
//
// Compile: nvcc -arch=compute_120a -code=sm_120a -o 14g_qmma_e3m2_e3m2_f32 14g_qmma_e3m2_e3m2_f32.cu
// Dump:    cuobjdump --dump-sass 14g_qmma_e3m2_e3m2_f32 > 14g_qmma_e3m2_e3m2_f32.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void qmma_e3m2_e3m2_f32_kernel(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
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
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32 "
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

    // Use zero inputs to get deterministic d[0] = 0 result. The SASS opcode
    // and control code are independent of input values.
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x00000000u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x00000000u;
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

    qmma_e3m2_e3m2_f32_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // With zero inputs, expected = 0.0 (sanity)
    printf("d[0] = %f (expected 0.0)\n", h_d[0]);

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
