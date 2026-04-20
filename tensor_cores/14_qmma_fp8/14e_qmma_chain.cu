// 14e_qmma_chain.cu
//
// Chapter 14 variant e. QMMA accumulator chaining analogue to kernel 13d for HMMA.
// Two QMMAs where D of the first feeds C of the second (accumulator in-place).
//
// Goal: validate that QMMA follows the same chaining patterns as HMMA:
//   - D and C colocated (in-place accumulator)
//   - A and B on distinct register blocks (re-read each QMMA)
//   - .reuse on B operand of all QMMAs except last
//   - 2 UIADD3 NOPs between QMMAs
//   - Control code bits 26 (SBS) and 27 (wait) for MMA scoreboard
//
// Both inputs E4M3 (the well-behaved case from 14a, avoids the FP4 layout gap).
// Each MMA accumulates 32 products of 1.0 * 1.0 = 32. Two chained = 64.0 expected.
//
// Compile: nvcc -arch=compute_120a -code=sm_120a -o 14e_qmma_chain 14e_qmma_chain.cu
// Dump:    cuobjdump --dump-sass 14e_qmma_chain > 14e_qmma_chain.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void qmma_chain_kernel(
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

    // Two chained QMMAs. The accumulator D from the first becomes the C of the second.
    // ptxas should colocate D and C registers and emit a serial chain.
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1)
    );

    d[tid * 4 + 0] = c0;
    d[tid * 4 + 1] = c1;
    d[tid * 4 + 2] = c2;
    d[tid * 4 + 3] = c3;
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

    for (int i = 0; i < n_a; ++i) h_a[i] = 0x38383838u;  // E4M3 1.0
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x38383838u;
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

    qmma_chain_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // Two chained MMAs, each summing 32 products of 1.0 * 1.0
    // 0 + 32 + 32 = 64.0 expected
    printf("d[0] = %f (expected 64.0)\n", h_d[0]);

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
