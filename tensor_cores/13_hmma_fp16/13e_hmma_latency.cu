// 13e_hmma_latency.cu
//
// Chapter 13 variant e. Microbenchmark measuring HMMA latency on SM120.
//
// Design: chain N_HMMA serial MMAs. Each D becomes the C of the next.
// This forces ptxas to keep the chain serial (no reordering possible due
// to data dependency). Uses clock64() before and after the chain to
// measure total cycles.
//
// Parametrized by N_HMMA via -DN_HMMA=<n>. Default 16.
// Compile:
//   nvcc -arch=sm_120 -DN_HMMA=16 -o 13e_hmma_latency_16 13e_hmma_latency.cu
//   nvcc -arch=sm_120 -DN_HMMA=32 -o 13e_hmma_latency_32 13e_hmma_latency.cu
//   nvcc -arch=sm_120 -DN_HMMA=64 -o 13e_hmma_latency_64 13e_hmma_latency.cu
//
// Dump:
//   cuobjdump --dump-sass 13e_hmma_latency_16 > 13e_hmma_latency_16.sass
//
// Each thread reports total_cycles and total_cycles / N_HMMA.

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef N_HMMA
#define N_HMMA 16
#endif

// Expand N_HMMA copies of the MMA via preprocessor recursion.
// Each iteration: D = A * B + D (accumulator in-place).
#define HMMA_ONCE                                                                \
    asm volatile(                                                                \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                     \
        "{%0, %1, %2, %3}, "                                                     \
        "{%4, %5, %6, %7}, "                                                     \
        "{%8, %9}, "                                                             \
        "{%0, %1, %2, %3};\n"                                                    \
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)                                 \
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),                                   \
           "r"(b0), "r"(b1)                                                      \
    );

__global__ void hmma_latency_kernel(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    const float*    __restrict__ c_init,
    float*          __restrict__ d,
    unsigned long long* __restrict__ cycles_out)
{
    const unsigned tid = threadIdx.x;

    // Load fragments (setup, not measured).
    uint32_t a0 = a[tid * 4 + 0];
    uint32_t a1 = a[tid * 4 + 1];
    uint32_t a2 = a[tid * 4 + 2];
    uint32_t a3 = a[tid * 4 + 3];

    uint32_t b0 = b[tid * 2 + 0];
    uint32_t b1 = b[tid * 2 + 1];

    float d0 = c_init[tid * 4 + 0];
    float d1 = c_init[tid * 4 + 1];
    float d2 = c_init[tid * 4 + 2];
    float d3 = c_init[tid * 4 + 3];

    // Fence to ensure all setup is done before timing.
    asm volatile("" ::: "memory");

    unsigned long long t0 = clock64();

    // The chained HMMAs. N_HMMA iterations, unrolled.
    #pragma unroll
    for (int i = 0; i < N_HMMA; ++i) {
        HMMA_ONCE
    }

    unsigned long long t1 = clock64();

    // Store results. The cycles count is stored only from thread 0 (whole warp is uniform).
    d[tid * 4 + 0] = d0;
    d[tid * 4 + 1] = d1;
    d[tid * 4 + 2] = d2;
    d[tid * 4 + 3] = d3;

    if (tid == 0) {
        cycles_out[0] = t1 - t0;
    }
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
    unsigned long long h_cycles[1] = {0};

    for (int i = 0; i < n_a; ++i) h_a[i] = 0x3c003c00u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x3c003c00u;
    for (int i = 0; i < n_c; ++i) h_c[i] = 0.0f;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0.0f;

    uint32_t *d_a, *d_b;
    float    *d_c, *d_d;
    unsigned long long *d_cycles;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t));
    cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_c, n_c * sizeof(float));
    cudaMalloc(&d_d, n_d * sizeof(float));
    cudaMalloc(&d_cycles, sizeof(unsigned long long));

    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n_c * sizeof(float),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, n_d * sizeof(float),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_cycles, h_cycles, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // Warmup run (eliminates first-launch effects).
    hmma_latency_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d, d_cycles);
    cudaDeviceSynchronize();

    // Timed run.
    hmma_latency_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d, d_cycles);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("N_HMMA       = %d\n", N_HMMA);
    printf("d[0]         = %f (expected %f)\n", h_d[0], (float)(16 * N_HMMA));
    printf("total cycles = %llu\n", h_cycles[0]);
    printf("cycles/HMMA  = %.2f\n", (double)h_cycles[0] / N_HMMA);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_cycles);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    return 0;
}
