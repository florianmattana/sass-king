// 13d_hmma_chain.cu
//
// Chapter 13 variant d. Accumulator chaining: two consecutive HMMAs where
// the D of the first feeds the C of the second. Same A and B for both MMAs
// to isolate the chaining pattern.
//
// Base: 13b template (FP16 input, FP32 accumulator).
//
// Source pattern:
//   mma.sync D1, A, B, C        // D1 = A*B + C
//   mma.sync D2, A, B, D1       // D2 = A*B + D1 = 2*A*B + C
//
// Expected D2 element = 2 * 16.0 = 32.0 (two accumulations of 1.0 * 1.0 * 16)
//
// Matches two invocations of CUTLASS SM80_16x8x16_F32F16F16F32_TN.
//
// Compile: nvcc -arch=sm_120 -o 13d_hmma_chain 13d_hmma_chain.cu
// Dump:    cuobjdump --dump-sass 13d_hmma_chain > 13d_hmma_chain.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void hmma_chain_kernel(
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

    // Intermediate accumulator after first MMA.
    float d0_1, d1_1, d2_1, d3_1;

    // Final accumulator after second MMA.
    float d0, d1, d2, d3;

    // First MMA: D1 = A * B + C
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0_1), "=f"(d1_1), "=f"(d2_1), "=f"(d3_1)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1),
           "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );

    // Second MMA: D2 = A * B + D1
    // Note: D1 is passed as C input to the second MMA.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1),
           "f"(d0_1), "f"(d1_1), "f"(d2_1), "f"(d3_1)
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

    for (int i = 0; i < n_a; ++i) h_a[i] = 0x3c003c00u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x3c003c00u;
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

    hmma_chain_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: 32.0 (two accumulations of 16.0)
    printf("d[0] = %f (expected 32.0)\n", h_d[0]);

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
