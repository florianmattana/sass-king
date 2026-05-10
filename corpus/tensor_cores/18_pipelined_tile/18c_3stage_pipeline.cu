// 18c_3stage_pipeline.cu
//
// Chapter 18 variant c. 3-stage software pipeline: 3 tiles prefetched
// simultaneously, then consumed one by one with DEPBAR.LE N at N = 2, 1, 0.
//
// Structure:
//   Prologue: prefetch tile[0], tile[1], tile[2]
//   Iter 0: wait tile[0] (N=2 in flight), LDSM + HMMA
//   Iter 1: wait tile[1] (N=1), LDSM + HMMA
//   Iter 2: wait tile[2] (N=0), LDSM + HMMA
//   Epilogue: STG
//
// 3 shared memory buffers (A and B each), full unroll.
//
// Compile: nvcc -arch=sm_120 -o 18c_3stage_pipeline 18c_3stage_pipeline.cu
// Dump:    cuobjdump --dump-sass 18c_3stage_pipeline > 18c_3stage_pipeline.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

// Macro to consume tile k from buffer b
#define CONSUME_TILE(buf) \
do { \
    unsigned row_a  = tid % 8; \
    unsigned tile_a = tid / 8; \
    __half* ptr_a = &smem_a[buf][tile_a * 64 + row_a * 8]; \
    uint32_t addr_a = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_a)); \
    unsigned lane = tid % 16; \
    unsigned row_b = lane % 8; \
    unsigned tile_b = lane / 8; \
    __half* ptr_b = &smem_b[buf][tile_b * 64 + row_b * 8]; \
    uint32_t addr_b = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_b)); \
    uint32_t a0, a1, a2, a3, b0, b1; \
    asm volatile( \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr_a)); \
    asm volatile( \
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(b0), "=r"(b1) : "r"(addr_b)); \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%10, %11, %12, %13};\n" \
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) \
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
           "r"(b0), "r"(b1), \
           "f"(d0), "f"(d1), "f"(d2), "f"(d3)); \
} while (0)

__global__ void three_stage_kernel(
    const __half* __restrict__ a_in,   // 3 * 256 halves
    const __half* __restrict__ b_in,   // 3 * 128 halves
    float*        __restrict__ d_out)
{
    __shared__ __half smem_a[3][TILE_A_HALVES];
    __shared__ __half smem_b[3][TILE_B_HALVES];

    const unsigned tid = threadIdx.x;

    uint32_t smem_a_ptr[3], smem_b_ptr[3];
    smem_a_ptr[0] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_a[0][0]));
    smem_a_ptr[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_a[1][0]));
    smem_a_ptr[2] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_a[2][0]));
    smem_b_ptr[0] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[0][0]));
    smem_b_ptr[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[1][0]));
    smem_b_ptr[2] = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_b[2][0]));

    // Prologue: prefetch all 3 tiles, one commit group each
    cp_async_16B(smem_a_ptr[0] + tid * 16, a_in + tid * 8);
    if (tid < 16) cp_async_16B(smem_b_ptr[0] + tid * 16, b_in + tid * 8);
    cp_async_commit();

    cp_async_16B(smem_a_ptr[1] + tid * 16, a_in + TILE_A_HALVES + tid * 8);
    if (tid < 16) cp_async_16B(smem_b_ptr[1] + tid * 16, b_in + TILE_B_HALVES + tid * 8);
    cp_async_commit();

    cp_async_16B(smem_a_ptr[2] + tid * 16, a_in + 2 * TILE_A_HALVES + tid * 8);
    if (tid < 16) cp_async_16B(smem_b_ptr[2] + tid * 16, b_in + 2 * TILE_B_HALVES + tid * 8);
    cp_async_commit();

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // Iter 0: wait tile 0 (2 still in flight)
    cp_async_wait<2>();
    __syncthreads();
    CONSUME_TILE(0);

    // Iter 1: wait tile 1 (1 still in flight)
    cp_async_wait<1>();
    __syncthreads();
    CONSUME_TILE(1);

    // Iter 2: wait tile 2 (0 still in flight)
    cp_async_wait<0>();
    __syncthreads();
    CONSUME_TILE(2);

    d_out[tid * 4 + 0] = d0;
    d_out[tid * 4 + 1] = d1;
    d_out[tid * 4 + 2] = d2;
    d_out[tid * 4 + 3] = d3;
}

int main(void) {
    const int n_a = 3 * TILE_A_HALVES;
    const int n_b = 3 * TILE_B_HALVES;
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

    three_stage_kernel<<<1, 32>>>(d_a, d_b, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // 3 tiles of m16n8k16 * 1.0 = 48
    printf("d[0] = %f (expected 48.0)\n", h_d[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_d);
    return 0;
}
