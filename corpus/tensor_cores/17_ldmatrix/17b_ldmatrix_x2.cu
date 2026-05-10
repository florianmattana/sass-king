// 17b_ldmatrix_x2.cu
//
// Chapter 17 variant b. ldmatrix.x2: load TWO 8x8 half-precision matrix tiles
// from shared memory into two uint32 registers per thread.
//
// Goal: identify how the x2 variant is encoded in SASS (mnemonic suffix,
// opcode bytes, or control code) by comparison with 17a.
//
// Matches CUTLASS SM75_U32x2_LDSM_N atom from copy_sm75.hpp (line 100-117).
//
// Warp layout for ldmatrix.x2.m8n8:
//   32 threads load 2 tiles of 8x8 = 128 half values total
//   Each thread receives 2 uint32 = 4 halves (2 per tile)
//   Lanes 0-7 supply addresses for tile 1
//   Lanes 8-15 supply addresses for tile 2
//   Lanes 16-31 are ignored
//
// Compile: nvcc -arch=sm_120 -o 17b_ldmatrix_x2 17b_ldmatrix_x2.cu
// Dump:    cuobjdump --dump-sass 17b_ldmatrix_x2 > 17b_ldmatrix_x2.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void ldmatrix_x2_kernel(const __half* __restrict__ in, uint32_t* __restrict__ out)
{
    // Shared memory: 128 halves (two 8x8 tiles), 256 bytes
    __shared__ __half smem[128];

    const unsigned tid = threadIdx.x;

    // Stage A: every thread writes 4 halves (128 halves / 32 threads = 4 per thread)
    smem[tid * 4 + 0] = in[tid * 4 + 0];
    smem[tid * 4 + 1] = in[tid * 4 + 1];
    smem[tid * 4 + 2] = in[tid * 4 + 2];
    smem[tid * 4 + 3] = in[tid * 4 + 3];

    __syncthreads();

    // Stage B: ldmatrix.x2
    // The address calculation: lanes 0-7 point to rows of tile 1 (smem[0..63]),
    // lanes 8-15 point to rows of tile 2 (smem[64..127]).
    // We compute a per-lane address that covers both patterns:
    //   lane = tid % 16
    //   row = lane % 8
    //   tile = lane / 8  (0 for first 8 lanes, 1 for next 8)
    //   offset_in_halves = tile * 64 + row * 8
    unsigned lane = tid % 16;
    unsigned row = lane % 8;
    unsigned tile = lane / 8;
    __half* ptr = &smem[tile * 64 + row * 8];

    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    uint32_t d0, d1;
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(d0), "=r"(d1)
        :  "r"(smem_int_ptr)
    );

    out[tid * 2 + 0] = d0;
    out[tid * 2 + 1] = d1;
}

int main(void) {
    const int n = 128;

    __half   *h_in  = (__half*)  malloc(n * sizeof(__half));
    uint32_t *h_out = (uint32_t*)malloc(64 * sizeof(uint32_t));

    for (int i = 0; i < n; ++i) {
        h_in[i] = __float2half((float)i);
    }

    __half   *d_in;
    uint32_t *d_out;
    cudaMalloc(&d_in,  n * sizeof(__half));
    cudaMalloc(&d_out, 64 * sizeof(uint32_t));

    cudaMemcpy(d_in, h_in, n * sizeof(__half), cudaMemcpyHostToDevice);

    ldmatrix_x2_kernel<<<1, 32>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 64 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Print first 4 thread results (each has 2 uint32 = 2 halves of tile 1 + 2 halves of tile 2)
    printf("First 4 thread outputs (each has d0=tile1, d1=tile2):\n");
    for (int i = 0; i < 4; ++i) {
        uint32_t d0 = h_out[i * 2 + 0];
        uint32_t d1 = h_out[i * 2 + 1];

        unsigned d0_lo = d0 & 0xffff;
        unsigned d0_hi = (d0 >> 16) & 0xffff;
        unsigned d1_lo = d1 & 0xffff;
        unsigned d1_hi = (d1 >> 16) & 0xffff;

        float d0_lo_f = __half2float(*((__half*)&d0_lo));
        float d0_hi_f = __half2float(*((__half*)&d0_hi));
        float d1_lo_f = __half2float(*((__half*)&d1_lo));
        float d1_hi_f = __half2float(*((__half*)&d1_hi));

        printf("  thread %2d: d0=0x%08x (%.1f, %.1f)  d1=0x%08x (%.1f, %.1f)\n",
               i, d0, d0_lo_f, d0_hi_f, d1, d1_lo_f, d1_hi_f);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
