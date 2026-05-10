// 17d_ldmatrix_x4_trans.cu
//
// Chapter 17 variant d. ldmatrix.x4.trans: load FOUR 8x8 half tiles from
// shared memory in TRANSPOSED form into four uint32 registers per thread.
//
// Goal: isolate the encoding of the .trans modifier by comparing with 17c
// (same kernel structure, only .trans added to the PTX).
//
// Matches CUTLASS SM75_U32x4_LDSM_T atom (equivalent to U32x4_LDSM_N but with
// .trans in the PTX).
//
// Same warp layout as 17c (32 lanes, each supplies its own address for one
// of 4 tiles), but the hardware delivers the values in transposed order.
//
// Compile: nvcc -arch=sm_120 -o 17d_ldmatrix_x4_trans 17d_ldmatrix_x4_trans.cu
// Dump:    cuobjdump --dump-sass 17d_ldmatrix_x4_trans > 17d_ldmatrix_x4_trans.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void ldmatrix_x4_trans_kernel(const __half* __restrict__ in, uint32_t* __restrict__ out)
{
    __shared__ __half smem[256];

    const unsigned tid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        smem[tid * 8 + i] = in[tid * 8 + i];
    }

    __syncthreads();

    // Same address calculation as 17c
    unsigned row = tid % 8;
    unsigned tile = tid / 8;
    __half* ptr = &smem[tile * 64 + row * 8];

    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    uint32_t d0, d1, d2, d3;
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        :  "r"(smem_int_ptr)
    );

    out[tid * 4 + 0] = d0;
    out[tid * 4 + 1] = d1;
    out[tid * 4 + 2] = d2;
    out[tid * 4 + 3] = d3;
}

int main(void) {
    const int n = 256;

    __half   *h_in  = (__half*)  malloc(n * sizeof(__half));
    uint32_t *h_out = (uint32_t*)malloc(128 * sizeof(uint32_t));

    for (int i = 0; i < n; ++i) {
        h_in[i] = __float2half((float)i);
    }

    __half   *d_in;
    uint32_t *d_out;
    cudaMalloc(&d_in,  n * sizeof(__half));
    cudaMalloc(&d_out, 128 * sizeof(uint32_t));

    cudaMemcpy(d_in, h_in, n * sizeof(__half), cudaMemcpyHostToDevice);

    ldmatrix_x4_trans_kernel<<<1, 32>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("First 4 thread outputs (x4.trans, expect column-ordered values):\n");
    for (int i = 0; i < 4; ++i) {
        uint32_t d0 = h_out[i * 4 + 0];
        uint32_t d1 = h_out[i * 4 + 1];
        uint32_t d2 = h_out[i * 4 + 2];
        uint32_t d3 = h_out[i * 4 + 3];

        auto decode = [](uint32_t v) -> std::pair<float,float> {
            unsigned lo = v & 0xffff;
            unsigned hi = (v >> 16) & 0xffff;
            return {__half2float(*((__half*)&lo)), __half2float(*((__half*)&hi))};
        };

        auto [d0_lo, d0_hi] = decode(d0);
        auto [d1_lo, d1_hi] = decode(d1);
        auto [d2_lo, d2_hi] = decode(d2);
        auto [d3_lo, d3_hi] = decode(d3);

        printf("  thread %2d: d0=(%.1f,%.1f) d1=(%.1f,%.1f) d2=(%.1f,%.1f) d3=(%.1f,%.1f)\n",
               i, d0_lo, d0_hi, d1_lo, d1_hi, d2_lo, d2_hi, d3_lo, d3_hi);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
