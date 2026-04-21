// 18a_pipelined_tile.cu
//
// Chapter 18 variant a. 2-stage software pipelined MMA tile using cp.async.
// This is the basic production pattern: prefetch next tile from global to shared
// memory while computing the current tile's MMA, hiding the memory latency.
//
// Pipeline structure:
//   Prologue:     cp.async tile[0] -> smem[0], fence, cp.async tile[1] -> smem[1], fence
//   Iteration 0:  wait for tile[0], LDSM+HMMA tile[0]
//   Iteration 1:  wait for tile[1], LDSM+HMMA tile[1]
//   Epilogue:     STG results
//
// The kernel computes a single m16n8k16 MMA with K loop over 2 tiles (k_total = 32).
// Simple enough to stay readable in SASS but captures the full pipelined pattern.
//
// For simplicity: A = 16x32 half, B = 32x8 half, D = 16x8 float.
// Tiles: A_tile = 16x16 half (256 halves), B_tile = 16x8 half (128 halves).
// K loop iterates 2 times with tile size 16 on K dimension.
//
// Compile: nvcc -arch=sm_120 -o 18a_pipelined_tile 18a_pipelined_tile.cu
// Dump:    cuobjdump --dump-sass 18a_pipelined_tile > 18a_pipelined_tile.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

__global__ void pipelined_tile_kernel(
    const __half* __restrict__ a_in,   // 16x32 half
    const __half* __restrict__ b_in,   // 32x8 half
    float*        __restrict__ d_out)  // 16x8 float
{
    // Double-buffered shared memory: 2 buffers for A (each 16x16 = 256 halves = 512 bytes)
    //                                 2 buffers for B (each 16x8 = 128 halves = 256 bytes)
    __shared__ __half smem_a[2][256];
    __shared__ __half smem_b[2][128];

    const unsigned tid = threadIdx.x;

    // Compute shared memory pointers for both buffers
    uint32_t smem_a_ptr[2];
    uint32_t smem_b_ptr[2];
    smem_a_ptr[0] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_a[0][0]));
    smem_a_ptr[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_a[1][0]));
    smem_b_ptr[0] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[0][0]));
    smem_b_ptr[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[1][0]));

    // Each thread copies 8 halves (16 bytes) of A and 4 halves (8 bytes) of B per tile
    // For cp.async we need 16-byte granularity, so we do 8 halves at a time.
    // A tile: 256 halves / 32 threads = 8 halves per thread = 16 bytes = 1 cp.async per thread
    // B tile: 128 halves / 32 threads = 4 halves per thread = 8 bytes, so we need cp.async of 8 bytes
    //         OR we let 16 threads each copy 8 halves of B

    // Stage 0: Prefetch tile 0 into buffer 0
    // A tile 0: global offset 0, each thread copies 16 bytes
    cp_async_16B(smem_a_ptr[0] + tid * 16, a_in + tid * 8);
    // B tile 0: 128 halves total, 16 threads each copy 16 bytes (8 halves), rest idle
    if (tid < 16) {
        cp_async_16B(smem_b_ptr[0] + tid * 16, b_in + tid * 8);
    }
    cp_async_commit();

    // Stage 1: Prefetch tile 1 into buffer 1
    // A tile 1: global offset 16 (K tile is 16x16 = 256 halves, next tile starts at col 16)
    // Actually A is 16 rows x 32 cols. Tile 0 = cols 0-15, tile 1 = cols 16-31.
    // Each row is 32 halves. Tile strides: thread tid row = tid/2, col offset = (tid%2)*8
    // Simpler: linearize. A has 16*32=512 halves. Tile 0 halves = linear 0..255 in transposed layout?
    // For simplicity we treat it as two contiguous blocks of 256 halves each.
    cp_async_16B(smem_a_ptr[1] + tid * 16, a_in + 256 + tid * 8);
    if (tid < 16) {
        cp_async_16B(smem_b_ptr[1] + tid * 16, b_in + 128 + tid * 8);
    }
    cp_async_commit();

    // Accumulator initialized to zero
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // Iteration 0: wait for tile 0, LDSM + HMMA
    cp_async_wait<1>();  // wait until only 1 group pending (tile 0 must be ready)
    __syncthreads();

    {
        unsigned row_a = tid % 8;
        unsigned tile_a = tid / 8;
        __half* ptr_a = &smem_a[0][tile_a * 64 + row_a * 8];
        uint32_t addr_a = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_a));

        unsigned lane = tid % 16;
        unsigned row_b = lane % 8;
        unsigned tile_b = lane / 8;
        __half* ptr_b = &smem_b[0][tile_b * 64 + row_b * 8];
        uint32_t addr_b = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_b));

        uint32_t a0, a1, a2, a3, b0, b1;
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            :  "r"(addr_a)
        );
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1)
            :  "r"(addr_b)
        );

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
               "r"(b0), "r"(b1),
               "f"(d0), "f"(d1), "f"(d2), "f"(d3)
        );
    }

    // Iteration 1: wait for tile 1, LDSM + HMMA
    cp_async_wait<0>();  // wait for all remaining
    __syncthreads();

    {
        unsigned row_a = tid % 8;
        unsigned tile_a = tid / 8;
        __half* ptr_a = &smem_a[1][tile_a * 64 + row_a * 8];
        uint32_t addr_a = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_a));

        unsigned lane = tid % 16;
        unsigned row_b = lane % 8;
        unsigned tile_b = lane / 8;
        __half* ptr_b = &smem_b[1][tile_b * 64 + row_b * 8];
        uint32_t addr_b = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_b));

        uint32_t a0, a1, a2, a3, b0, b1;
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            :  "r"(addr_a)
        );
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1)
            :  "r"(addr_b)
        );

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
               "r"(b0), "r"(b1),
               "f"(d0), "f"(d1), "f"(d2), "f"(d3)
        );
    }

    d_out[tid * 4 + 0] = d0;
    d_out[tid * 4 + 1] = d1;
    d_out[tid * 4 + 2] = d2;
    d_out[tid * 4 + 3] = d3;
}

int main(void) {
    const int n_a = 512;   // 16x32 halves
    const int n_b = 256;   // 32x8 halves
    const int n_d = 128;   // 16x8 floats

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

    pipelined_tile_kernel<<<1, 32>>>(d_a, d_b, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // Two tiles of m16n8k16 with all 1.0 inputs: each d element = 2 * 16 * (1*1) = 32.0
    printf("d[0] = %f (expected 32.0)\n", h_d[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_d);
    return 0;
}
