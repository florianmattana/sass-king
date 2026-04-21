// 17f_ldmatrix_latency.cu
//
// Chapter 17 variant f. LDSM serial latency microbenchmark. Analogue to 13e/14f
// for HMMA/QMMA. N chained LDSM.x1 where each LDSM's destination is used as part
// of the next LDSM's address, forcing serial dependency.
//
// Trick: fill shared memory with 0x0000 halves so every LDSM returns 0. The
// dependency chain is real (compiler sees d feed into addr) but the address
// computed stays valid (base_addr + 0 = base_addr).
//
// Compile with N=16, 32, 64:
//   nvcc -arch=sm_120 -DN_LDSM=16 -o 17f_ldmatrix_latency_16 17f_ldmatrix_latency.cu
//   nvcc -arch=sm_120 -DN_LDSM=32 -o 17f_ldmatrix_latency_32 17f_ldmatrix_latency.cu
//   nvcc -arch=sm_120 -DN_LDSM=64 -o 17f_ldmatrix_latency_64 17f_ldmatrix_latency.cu

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef N_LDSM
#define N_LDSM 16
#endif

// One chained LDSM: input addr %1 (uint32 smem ptr), modified by d result
// The trick: d will always be 0 because smem content is 0, so addr stays valid
#define ONE_LDSM \
    "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
    "add.u32 %1, %1, %0;\n"

#define LDSM_2   ONE_LDSM ONE_LDSM
#define LDSM_4   LDSM_2 LDSM_2
#define LDSM_8   LDSM_4 LDSM_4
#define LDSM_16  LDSM_8 LDSM_8
#define LDSM_32  LDSM_16 LDSM_16
#define LDSM_64  LDSM_32 LDSM_32

#if N_LDSM == 16
#define LDSM_CHAIN LDSM_16
#elif N_LDSM == 32
#define LDSM_CHAIN LDSM_32
#elif N_LDSM == 64
#define LDSM_CHAIN LDSM_64
#else
#error "N_LDSM must be 16, 32, or 64"
#endif

__global__ void ldmatrix_latency_kernel(
    uint32_t* __restrict__ out,
    unsigned long long* __restrict__ cycles_out)
{
    __shared__ __half smem[64];

    const unsigned tid = threadIdx.x;

    // Fill shared memory with zeros (halves 0x0000)
    smem[tid * 2 + 0] = __float2half(0.0f);
    smem[tid * 2 + 1] = __float2half(0.0f);

    __syncthreads();

    // Initial address for lane
    unsigned row = tid % 8;
    __half* ptr = &smem[row * 8];
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    uint32_t d;

    unsigned long long start = clock64();

    asm volatile(
        LDSM_CHAIN
        : "=r"(d), "+r"(addr)
        :
    );

    unsigned long long end = clock64();

    out[tid] = d;

    if (tid == 0) {
        *cycles_out = end - start;
    }
}

int main(void) {
    uint32_t           *d_out;
    unsigned long long *d_cycles;
    cudaMalloc(&d_out,    32 * sizeof(uint32_t));
    cudaMalloc(&d_cycles, sizeof(unsigned long long));

    ldmatrix_latency_kernel<<<1, 32>>>(d_out, d_cycles);
    cudaDeviceSynchronize();

    uint32_t           h_out[32];
    unsigned long long h_cycles = 0;
    cudaMemcpy(h_out,     d_out,    32 * sizeof(uint32_t),   cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("N=%d  total_cycles=%llu  cycles_per_ldsm=%.2f  d[0]=0x%08x (expected 0x00000000)\n",
           N_LDSM, h_cycles, (double)h_cycles / N_LDSM, h_out[0]);

    cudaFree(d_out);
    cudaFree(d_cycles);
    return 0;
}
