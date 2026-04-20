// 14j_qmma_e4m3_e4m3_f16.cu
//
// Chapter 14 variant j. QMMA with E4M3 inputs and FP16 accumulator (vs FP32 in 14a).
// Purpose: identify the bits of the QMMA control code that encode the accumulator
// dtype (F16 vs F32). Compare to 14a to isolate this delta.
//
// Matches CUTLASS SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, half_t>
// from mma_sm120.hpp (line 1524).
//
// Fragment layout per thread (different from 14a):
//   A: uint32_t[4] = 16 e4m3 per thread (same as 14a)
//   B: uint32_t[2] = 8 e4m3 per thread (same as 14a)
//   C: uint32_t[2] = 4 half per thread (HALVED vs 14a)
//   D: uint32_t[2] = 4 half per thread (HALVED vs 14a)
//
// Note: with k=32 and inputs 1.0 each, FP16 accumulator stays well within range
// (32.0 << FP16 max ~65504).
//
// Compile: nvcc -arch=compute_120a -code=sm_120a -o 14j_qmma_e4m3_e4m3_f16 14j_qmma_e4m3_e4m3_f16.cu
// Dump:    cuobjdump --dump-sass 14j_qmma_e4m3_e4m3_f16 > 14j_qmma_e4m3_e4m3_f16.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void qmma_e4m3_e4m3_f16_kernel(
    const uint32_t* __restrict__ a,  // A: E4M3
    const uint32_t* __restrict__ b,  // B: E4M3
    const uint32_t* __restrict__ c,  // C: FP16 packed (2 half per uint32)
    uint32_t*       __restrict__ d)  // D: FP16 packed
{
    const unsigned tid = threadIdx.x;

    uint32_t a0 = a[tid * 4 + 0];
    uint32_t a1 = a[tid * 4 + 1];
    uint32_t a2 = a[tid * 4 + 2];
    uint32_t a3 = a[tid * 4 + 3];

    uint32_t b0 = b[tid * 2 + 0];
    uint32_t b1 = b[tid * 2 + 1];

    uint32_t c0 = c[tid * 2 + 0];
    uint32_t c1 = c[tid * 2 + 1];

    uint32_t d0, d1;

    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
        : "=r"(d0), "=r"(d1)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1),
           "r"(c0), "r"(c1)
    );

    d[tid * 2 + 0] = d0;
    d[tid * 2 + 1] = d1;
}

int main(void) {
    const int n_a = 128;
    const int n_b = 64;
    const int n_c = 64;   // half count of 14a (FP16 packed)
    const int n_d = 64;

    uint32_t *h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    uint32_t *h_c = (uint32_t*)malloc(n_c * sizeof(uint32_t));
    uint32_t *h_d = (uint32_t*)malloc(n_d * sizeof(uint32_t));

    // E4M3 1.0 = 0x38, packed 0x38383838
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x38383838u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x38383838u;
    // C = 0 (two FP16 zeros packed = 0x00000000)
    for (int i = 0; i < n_c; ++i) h_c[i] = 0x00000000u;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0x00000000u;

    uint32_t *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t));
    cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_c, n_c * sizeof(uint32_t));
    cudaMalloc(&d_d, n_d * sizeof(uint32_t));

    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n_c * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, n_d * sizeof(uint32_t), cudaMemcpyHostToDevice);

    qmma_e4m3_e4m3_f16_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Each D element = sum of 32 products of 1.0 * 1.0 = 32.0
    // In FP16, 32.0 = 0x5000. Packed two halves = 0x50005000.
    printf("d[0] = 0x%08x (expected 0x50005000 = two FP16 32.0)\n", h_d[0]);

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
