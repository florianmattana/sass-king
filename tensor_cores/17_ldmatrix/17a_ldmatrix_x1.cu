// 17a_ldmatrix_x1.cu
//
// Chapter 17 variant a. Minimal ldmatrix test: load one 8x8 half-precision
// matrix tile from shared memory into a single uint32 register per thread.
//
// Goal: identify the SASS opcode emitted for
//   ldmatrix.sync.aligned.x1.m8n8.shared.b16
// and establish the base case of the ldmatrix family on SM120.
//
// Matches CUTLASS SM75_U32x1_LDSM_N atom from copy_sm75.hpp (line 80-97).
//
// Warp layout for ldmatrix.x1.m8n8:
//   32 threads load 1 tile of 8x8 = 64 half values
//   Each thread receives 2 halves (packed in 1 uint32)
//   The 8 threads with lane_id in [0, 7] provide the row pointers
//   The other 24 threads' address input is ignored
//
// Compile: nvcc -arch=sm_120 -o 17a_ldmatrix_x1 17a_ldmatrix_x1.cu
// Dump:    cuobjdump --dump-sass 17a_ldmatrix_x1 > 17a_ldmatrix_x1.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void ldmatrix_x1_kernel(const __half* __restrict__ in, uint32_t* __restrict__ out)
{
    // Shared memory: 64 halves (one 8x8 tile), 128 bytes
    __shared__ __half smem[64];

    const unsigned tid = threadIdx.x;

    // Stage A: every thread writes its share of the tile
    // We keep it simple: thread tid writes smem[tid * 2] and smem[tid * 2 + 1]
    // from input array.
    smem[tid * 2 + 0] = in[tid * 2 + 0];
    smem[tid * 2 + 1] = in[tid * 2 + 1];

    __syncthreads();

    // Stage B: the ldmatrix instruction
    // Each of the 8 "row threads" (lanes 0-7) provides the address of its row.
    // Other threads supply addresses that the hardware ignores.
    // For a simple pattern, we have each thread point to smem + (tid % 8) * 8 halves
    // so lanes 0-7 point to consecutive rows.
    unsigned row = tid % 8;
    __half* row_ptr = &smem[row * 8];

    // Convert the generic pointer to a shared memory pointer (uint32).
    // __cvta_generic_to_shared is the modern CUDA 11+ intrinsic.
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(row_ptr));

    uint32_t d;
    asm volatile(
        "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(d)
        :  "r"(smem_int_ptr)
    );

    out[tid] = d;
}

int main(void) {
    const int n = 64;

    __half   *h_in  = (__half*)  malloc(n * sizeof(__half));
    uint32_t *h_out = (uint32_t*)malloc(32 * sizeof(uint32_t));

    // Fill input with a recognizable pattern: half(tid) for position tid.
    // The packed uint32 for thread tid receiving two halves h0, h1 will be
    // (h1_bits << 16) | h0_bits.
    for (int i = 0; i < n; ++i) {
        h_in[i] = __float2half((float)i);
    }

    __half   *d_in;
    uint32_t *d_out;
    cudaMalloc(&d_in,  n * sizeof(__half));
    cudaMalloc(&d_out, 32 * sizeof(uint32_t));

    cudaMemcpy(d_in, h_in, n * sizeof(__half), cudaMemcpyHostToDevice);

    ldmatrix_x1_kernel<<<1, 32>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Print first 8 thread results (the 8 matrix rows loaded)
    printf("First 4 thread outputs (each uint32 = two halves packed):\n");
    for (int i = 0; i < 4; ++i) {
        unsigned lo = h_out[i] & 0xffff;
        unsigned hi = (h_out[i] >> 16) & 0xffff;
        float lo_f = __half2float(*((__half*)&lo));
        float hi_f = __half2float(*((__half*)&hi));
        printf("  thread %2d: 0x%08x  (halves: %.1f, %.1f)\n", i, h_out[i], lo_f, hi_f);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
