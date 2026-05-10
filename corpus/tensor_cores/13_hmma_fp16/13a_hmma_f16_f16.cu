// 13a_hmma_f16_f16.cu
//
// First tensor core chapter. Minimal HMMA at shape m16n8k16 with FP16 inputs
// and FP16 accumulator. Heritage SM80 atom valid on SM120.
//
// Goal: observe the SASS opcode emitted for mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// and establish the baseline for chapter 13. Compared against chapter 14 (QMMA FP8 kind::f8f6f4),
// chapter 16 (block-scaled), and further chapters.
//
// Matches CUTLASS SM80_16x8x16_F16F16F16F16_TN from include/cute/arch/mma_sm80.hpp (line 92).
//
// Fragment layout per thread (from CUTLASS):
//   A: uint32_t[4] = 8 half per thread (16x16 / 32 threads = 8 half/thread)
//   B: uint32_t[2] = 4 half per thread (16x8  / 32 threads = 4 half/thread)
//   C: uint32_t[2] = 4 half per thread (16x8  / 32 threads = 4 half/thread)
//   D: uint32_t[2] = 4 half per thread
//
// Warp-level: 32 threads cooperate on one 16x8 output tile.
//
// Compile: nvcc -arch=sm_120 -o 13a_hmma_f16_f16 13a_hmma_f16_f16.cu
// Dump:    cuobjdump --dump-sass 13a_hmma_f16_f16 > 13a_hmma_f16_f16.sass
#include<cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

__global__ void hmma_f16_f16_kernel(
    const uint32_t* __restrict__ a,  // 16x16 FP16 = 256 half = 128 uint32 total for the warp
    const uint32_t* __restrict__ b,  // 16x8  FP16 = 128 half =  64 uint32 total
    const uint32_t* __restrict__ c,  // 16x8  FP16 =  64 half =  32 uint32 total (accumulator input)
    uint32_t* __restrict__ d)        // 16x8  FP16 =  64 half =  32 uint32 total
{
    const unsigned tid = threadIdx.x;  // 0..31

    // Per-thread fragment inputs.
    // A: 4 uint32 per thread.
    uint32_t a0 = a[tid * 4 + 0];
    uint32_t a1 = a[tid * 4 + 1];
    uint32_t a2 = a[tid * 4 + 2];
    uint32_t a3 = a[tid * 4 + 3];

    // B: 2 uint32 per thread.
    uint32_t b0 = b[tid * 2 + 0];
    uint32_t b1 = b[tid * 2 + 1];

    // C: 2 uint32 per thread.
    uint32_t c0 = c[tid * 2 + 0];
    uint32_t c1 = c[tid * 2 + 1];

    // D accumulator output.
    uint32_t d0, d1;

    // The MMA under study. Exact form from CUTLASS SM80_16x8x16_F16F16F16F16_TN.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
        : "=r"(d0), "=r"(d1)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1),
           "r"(c0), "r"(c1)
    );

    // Store result.
    d[tid * 2 + 0] = d0;
    d[tid * 2 + 1] = d1;
}

int main(void) {
    // Sizes in uint32 units.
    const int n_a = 128;  // 16x16 half = 256 half = 128 uint32
    const int n_b = 64;   // 16x8  half = 128 half =  64 uint32
    const int n_c = 32;   // 16x8  half =  64 half =  32 uint32
    const int n_d = 32;

    uint32_t *h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    uint32_t *h_c = (uint32_t*)malloc(n_c * sizeof(uint32_t));
    uint32_t *h_d = (uint32_t*)malloc(n_d * sizeof(uint32_t));

    // Initialize all A and B elements to 1.0 (FP16 bit pattern 0x3c00).
    // Packed as two halves per uint32: 0x3c003c00.
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x3c003c00u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x3c003c00u;
    for (int i = 0; i < n_c; ++i) h_c[i] = 0x00000000u;  // C = 0
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

    // Launch one warp: 1 block, 32 threads.
    hmma_f16_f16_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Sanity: each D element should equal 16.0 (sum of 16 terms of 1.0 * 1.0).
    // 16.0 in FP16 is 0x4c00. Packed: 0x4c004c00.
    printf("d[0] = 0x%08x (expected 0x4c004c00 = two halves of 16.0)\n", h_d[0]);

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
