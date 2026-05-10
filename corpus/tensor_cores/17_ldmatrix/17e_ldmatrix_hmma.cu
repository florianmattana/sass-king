// 17e_ldmatrix_hmma.cu
//
// Chapter 17 variant e. Production pattern: load A fragment with LDSM.x4,
// load B fragment with LDSM.x2, feed directly into HMMA.16816.F32.
//
// This is the baseline GEMM tile: shared memory to tensor core via ldmatrix.
// Goal: identify the scoreboard pattern between LDSM and MMA, any NOPs needed,
// and how ptxas orchestrates register allocation across LDSM_dst -> MMA_src.
//
// HMMA.16816.F32 shape m16n8k16 requires:
//   A: 4 uint32 per thread = 8 halves per thread = 32 halves total in 8 lanes
//   B: 2 uint32 per thread = 4 halves per thread = 16 halves
//   C/D: float[4] accumulator per thread
//
// With shape 16x8x16 half*half = FP32:
//   A fragment is 16 rows x 16 cols = 256 halves = 4 tiles of 8x8
//   B fragment is 16 rows x 8 cols = 128 halves = 2 tiles of 8x8
//
// Compile: nvcc -arch=sm_120 -o 17e_ldmatrix_hmma 17e_ldmatrix_hmma.cu
// Dump:    cuobjdump --dump-sass 17e_ldmatrix_hmma > 17e_ldmatrix_hmma.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void ldmatrix_hmma_kernel(
    const __half* __restrict__ a_in,
    const __half* __restrict__ b_in,
    float*        __restrict__ d_out)
{
    __shared__ __half smem_a[256];  // 4 tiles of 8x8 for A (16x16 matrix)
    __shared__ __half smem_b[128];  // 2 tiles of 8x8 for B (16x8 matrix)

    const unsigned tid = threadIdx.x;

    // Stage 1: load A and B into shared memory
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        smem_a[tid * 8 + i] = a_in[tid * 8 + i];
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        smem_b[tid * 4 + i] = b_in[tid * 4 + i];
    }

    __syncthreads();

    // Stage 2: ldmatrix.x4 for A
    unsigned row_a = tid % 8;
    unsigned tile_a = tid / 8;
    __half* ptr_a = &smem_a[tile_a * 64 + row_a * 8];
    uint32_t smem_int_ptr_a = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_a));

    uint32_t a0, a1, a2, a3;
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        :  "r"(smem_int_ptr_a)
    );

    // Stage 3: ldmatrix.x2 for B
    unsigned lane = tid % 16;
    unsigned row_b = lane % 8;
    unsigned tile_b = lane / 8;
    __half* ptr_b = &smem_b[tile_b * 64 + row_b * 8];
    uint32_t smem_int_ptr_b = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_b));

    uint32_t b0, b1;
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b0), "=r"(b1)
        :  "r"(smem_int_ptr_b)
    );

    // Stage 4: HMMA.16816.F32
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1),
           "f"(d0), "f"(d1), "f"(d2), "f"(d3)
    );

    d_out[tid * 4 + 0] = d0;
    d_out[tid * 4 + 1] = d1;
    d_out[tid * 4 + 2] = d2;
    d_out[tid * 4 + 3] = d3;
}

int main(void) {
    const int n_a = 256;
    const int n_b = 128;
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

    ldmatrix_hmma_kernel<<<1, 32>>>(d_a, d_b, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // HMMA m16n8k16 with all 1.0 inputs: each d element = sum of 16 * (1*1) = 16
    printf("d[0] = %f (expected 16.0)\n", h_d[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_d);
    return 0;
}
