// 18b_kloop_pipeline.cu
//
// Chapter 18 variant b. K-loop 4 tiles with 2-stage software pipeline.
// Standard production GEMM pattern: prefetch next tile while computing current.
//
// Pipeline structure (software double-buffered):
//   Prologue:  cp.async tile[0] -> smem[0], commit
//   For k = 0 .. K-2:
//     cp.async tile[k+1] -> smem[(k+1) % 2], commit
//     wait for tile[k] ready
//     LDSM tile[k] from smem[k % 2]
//     HMMA with accumulator
//   Final k = K-1:
//     wait for tile[K-1]
//     LDSM + HMMA
//   Epilogue: STG
//
// Each tile is m16n8k16 half: A_tile = 16x16 halves = 256 halves = 512 bytes
//                              B_tile = 16x8 halves = 128 halves = 256 bytes
//
// K = 4 tiles, so full GEMM is m16n8k64 reduction.
//
// Compile: nvcc -arch=sm_120 -o 18b_kloop_pipeline 18b_kloop_pipeline.cu
// Dump:    cuobjdump --dump-sass 18b_kloop_pipeline > 18b_kloop_pipeline.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define K_TILES 4
#define TILE_A_HALVES 256
#define TILE_B_HALVES 128

__device__ __forceinline__ void cp_async_16B(uint32_t smem_int_ptr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__global__ void kloop_pipeline_kernel(
    const __half* __restrict__ a_in,   // K_TILES * 256 halves
    const __half* __restrict__ b_in,   // K_TILES * 128 halves
    float*        __restrict__ d_out)  // 16x8 float
{
    // Double-buffered shared memory
    __shared__ __half smem_a[2][TILE_A_HALVES];
    __shared__ __half smem_b[2][TILE_B_HALVES];

    const unsigned tid = threadIdx.x;

    uint32_t smem_a_ptr[2];
    uint32_t smem_b_ptr[2];
    smem_a_ptr[0] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_a[0][0]));
    smem_a_ptr[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_a[1][0]));
    smem_b_ptr[0] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[0][0]));
    smem_b_ptr[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[1][0]));

    // Prologue: issue cp.async for tile 0
    cp_async_16B(smem_a_ptr[0] + tid * 16, a_in + tid * 8);
    if (tid < 16) {
        cp_async_16B(smem_b_ptr[0] + tid * 16, b_in + tid * 8);
    }
    cp_async_commit();

    // Accumulator across all K tiles
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // Main loop: process tile k, prefetch tile k+1
    #pragma unroll 1  // force real loop, not unrolled
    for (int k = 0; k < K_TILES - 1; ++k) {
        int cur  = k & 1;
        int next = (k + 1) & 1;

        // Prefetch tile k+1
        cp_async_16B(smem_a_ptr[next] + tid * 16,
                     a_in + (k + 1) * TILE_A_HALVES + tid * 8);
        if (tid < 16) {
            cp_async_16B(smem_b_ptr[next] + tid * 16,
                         b_in + (k + 1) * TILE_B_HALVES + tid * 8);
        }
        cp_async_commit();

        // Wait for tile k (keep 1 group in flight = the prefetch we just issued)
        cp_async_wait<1>();
        __syncthreads();

        // LDSM + HMMA on tile k
        unsigned row_a  = tid % 8;
        unsigned tile_a = tid / 8;
        __half* ptr_a = &smem_a[cur][tile_a * 64 + row_a * 8];
        uint32_t addr_a = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_a));

        unsigned lane   = tid % 16;
        unsigned row_b  = lane % 8;
        unsigned tile_b = lane / 8;
        __half* ptr_b = &smem_b[cur][tile_b * 64 + row_b * 8];
        uint32_t addr_b = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_b));

        uint32_t a0, a1, a2, a3, b0, b1;
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr_a));
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1) : "r"(addr_b));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
               "r"(b0), "r"(b1),
               "f"(d0), "f"(d1), "f"(d2), "f"(d3));
    }

    // Tail: process last tile (no more prefetch)
    {
        int cur = (K_TILES - 1) & 1;

        cp_async_wait<0>();
        __syncthreads();

        unsigned row_a  = tid % 8;
        unsigned tile_a = tid / 8;
        __half* ptr_a = &smem_a[cur][tile_a * 64 + row_a * 8];
        uint32_t addr_a = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_a));

        unsigned lane   = tid % 16;
        unsigned row_b  = lane % 8;
        unsigned tile_b = lane / 8;
        __half* ptr_b = &smem_b[cur][tile_b * 64 + row_b * 8];
        uint32_t addr_b = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_b));

        uint32_t a0, a1, a2, a3, b0, b1;
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr_a));
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1) : "r"(addr_b));

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
               "r"(b0), "r"(b1),
               "f"(d0), "f"(d1), "f"(d2), "f"(d3));
    }

    d_out[tid * 4 + 0] = d0;
    d_out[tid * 4 + 1] = d1;
    d_out[tid * 4 + 2] = d2;
    d_out[tid * 4 + 3] = d3;
}

int main(void) {
    const int n_a = K_TILES * TILE_A_HALVES;   // 4 * 256 = 1024
    const int n_b = K_TILES * TILE_B_HALVES;   // 4 * 128 = 512
    const int n_d = 128;

    __half *h_a = (__half*)malloc(n_a * sizeof(__half));
    __half *h_b = (__half*)malloc(n_b * sizeof(__half));
    float  *h_d = (float*) malloc(n_d * sizeof(float));

    for (int i = 0; i < n_a; ++i) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < n_b; ++i) h_b[i] = __float2half(1.0f);
    for (int i = 0; i < n_d; ++i) h_d[i] = 0.0f;

    __half *d_a, *d_b;
    float  *d_d;
    cudaMalloc(&d_a, n_a * sizeof(__half));
    cudaMalloc(&d_b, n_b * sizeof(__half));
    cudaMalloc(&d_d, n_d * sizeof(float));

    cudaMemcpy(d_a, h_a, n_a * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(__half), cudaMemcpyHostToDevice);

    kloop_pipeline_kernel<<<1, 32>>>(d_a, d_b, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // 4 tiles of m16n8k16 with all 1.0: d = 4 * 16 = 64
    printf("d[0] = %f (expected 64.0)\n", h_d[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_d);
    return 0;
}
