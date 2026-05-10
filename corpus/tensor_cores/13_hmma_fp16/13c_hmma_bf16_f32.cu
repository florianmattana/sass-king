// 13c_hmma_bf16_f32.cu
//
// Chapter 13 variant c. Same shape (m16n8k16) and same accumulator (FP32) as 13b,
// but BF16 inputs instead of FP16. Isolates the SASS delta induced by BF16 vs FP16
// input dtype alone.
//
// Matches CUTLASS SM80_16x8x16_F32BF16BF16F32_TN from mma_sm80.hpp (line 224).
//
// Fragment layout per thread (from CUTLASS):
//   A: uint32_t[4] = 8 bf16 per thread (same register count as 13b FP16)
//   B: uint32_t[2] = 4 bf16 per thread
//   C: float[4]    = 4 float per thread (same as 13b)
//   D: float[4]
//
// BF16 bit pattern for 1.0 = 0x3f80. Packed two per uint32 = 0x3f803f80.
// Contrast with FP16 1.0 = 0x3c00, packed 0x3c003c00.
//
// Compile: nvcc -arch=sm_120 -o 13c_hmma_bf16_f32 13c_hmma_bf16_f32.cu
// Dump:    cuobjdump --dump-sass 13c_hmma_bf16_f32 > 13c_hmma_bf16_f32.sass

#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void hmma_bf16_f32_kernel(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    const float*    __restrict__ c,
    float*          __restrict__ d)
{
    const unsigned tid = threadIdx.x;

    uint32_t a0 = a[tid * 4 + 0];
    uint32_t a1 = a[tid * 4 + 1];
    uint32_t a2 = a[tid * 4 + 2];
    uint32_t a3 = a[tid * 4 + 3];

    uint32_t b0 = b[tid * 2 + 0];
    uint32_t b1 = b[tid * 2 + 1];

    float c0 = c[tid * 4 + 0];
    float c1 = c[tid * 4 + 1];
    float c2 = c[tid * 4 + 2];
    float c3 = c[tid * 4 + 3];

    float d0, d1, d2, d3;

    // The MMA under study. Exact form from CUTLASS SM80_16x8x16_F32BF16BF16F32_TN.
    // Only difference with 13b: "bf16" instead of "f16" for A and B dtype fields.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1),
           "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );

    d[tid * 4 + 0] = d0;
    d[tid * 4 + 1] = d1;
    d[tid * 4 + 2] = d2;
    d[tid * 4 + 3] = d3;
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

    // BF16 1.0 = 0x3f80. Packed two per uint32 = 0x3f803f80.
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x3f803f80u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x3f803f80u;
    for (int i = 0; i < n_c; ++i) h_c[i] = 0.0f;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0.0f;

    uint32_t *d_a, *d_b;
    float    *d_c, *d_d;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t));
    cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_c, n_c * sizeof(float));
    cudaMalloc(&d_d, n_d * sizeof(float));

    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n_c * sizeof(float),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, n_d * sizeof(float),    cudaMemcpyHostToDevice);

    hmma_bf16_f32_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: 16.0 (sum of 16 products of 1.0 * 1.0)
    printf("d[0] = %f (expected 16.0)\n", h_d[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    return 0;
}
