// 16d_omma_4x_latency.cu
//
// Chapter 16 variant d. Serial latency microbenchmark for OMMA.SF.16864 with
// scale_vec::4X and ue4m3 scales (the FP4 peak path, announced 900+ TFLOPS).
//
// Measures cycles-per-MMA by chaining N OMMAs where each MMA's C operand
// is the previous MMA's D output. This forces serialization: the tensor core
// cannot pipeline the chain, each MMA must wait for the previous to complete.
//
// Compiled with a template parameter N so the same binary has three variants
// (N = 16, 32, 64) at compile time. Three outputs let us fit a linear model:
//   total_cycles(N) = prologue + N * cycles_per_MMA
//
// Expected for comparison with other MMA families on SM120:
//   HMMA (chap 13e):    ~35 cycles/MMA serial
//   QMMA (chap 14f):    ~35 cycles/MMA serial
//   OMMA 4X (this):     ? cycles/MMA serial
//
// If the "900 TFLOPS" peak is realized via throughput (more FMAs per MMA thanks
// to k=64 with FP4 density), OMMA latency should be similar to HMMA/QMMA.
// If OMMA has a different latency (e.g. higher due to scale factor handling),
// the peak path tradeoff would be visible at this microbenchmark level.
//
// Compile (three separate binaries, one per N):
//   nvcc -arch=compute_120a -code=sm_120a -DN_MMA=16 -o 16d_omma_4x_latency_16 16d_omma_4x_latency.cu
//   nvcc -arch=compute_120a -code=sm_120a -DN_MMA=32 -o 16d_omma_4x_latency_32 16d_omma_4x_latency.cu
//   nvcc -arch=compute_120a -code=sm_120a -DN_MMA=64 -o 16d_omma_4x_latency_64 16d_omma_4x_latency.cu

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef N_MMA
#define N_MMA 16
#endif

__global__ void omma_4x_latency_kernel(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    float*          __restrict__ d,
    uint64_t*       __restrict__ cycles_out)
{
    const unsigned tid = threadIdx.x;

    uint32_t a0 = a[tid * 4 + 0];
    uint32_t a1 = a[tid * 4 + 1];
    uint32_t a2 = a[tid * 4 + 2];
    uint32_t a3 = a[tid * 4 + 3];

    uint32_t b0 = b[tid * 2 + 0];
    uint32_t b1 = b[tid * 2 + 1];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // Scale factors: 4X ue4m3, each scale = 1.0 (0x38), four packed = 0x38383838
    uint32_t sfa = 0x38383838u;
    uint32_t sfb = 0x38383838u;
    uint16_t bid = 0;
    uint16_t tid_sf = 0;

    // Barrier to align all warps before measurement
    __syncthreads();

    uint64_t t_start = clock64();

    // Chained OMMAs: D depends on previous D via C operand
    #pragma unroll
    for (int i = 0; i < N_MMA; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3}, "
            "{%10}, {%11, %12}, "
            "{%13}, {%14, %15};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
               "r"(b0), "r"(b1),
               "r"(sfa), "h"(bid), "h"(tid_sf),
               "r"(sfb), "h"(bid), "h"(tid_sf)
        );
    }

    uint64_t t_end = clock64();

    d[tid * 4 + 0] = d0;
    d[tid * 4 + 1] = d1;
    d[tid * 4 + 2] = d2;
    d[tid * 4 + 3] = d3;

    if (tid == 0) {
        cycles_out[0] = t_end - t_start;
    }
}

int main(void) {
    const int n_a = 128;
    const int n_b = 64;
    const int n_d = 128;

    uint32_t *h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    float    *h_d = (float*)   malloc(n_d * sizeof(float));
    uint64_t  h_cycles = 0;

    // Zero inputs (compute correctness not the goal here, SASS is)
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x00000000u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x00000000u;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0.0f;

    uint32_t *d_a, *d_b;
    float    *d_d;
    uint64_t *d_cycles;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t));
    cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_d, n_d * sizeof(float));
    cudaMalloc(&d_cycles, sizeof(uint64_t));

    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Warmup run
    omma_4x_latency_kernel<<<1, 32>>>(d_a, d_b, d_d, d_cycles);
    cudaDeviceSynchronize();

    // Measured run
    omma_4x_latency_kernel<<<1, 32>>>(d_a, d_b, d_d, d_cycles);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("N_MMA = %d, cycles = %lu, cycles/MMA = %.2f\n",
           N_MMA, (unsigned long)h_cycles, (double)h_cycles / (double)N_MMA);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    cudaFree(d_cycles);
    free(h_a);
    free(h_b);
    free(h_d);
    return 0;
}
