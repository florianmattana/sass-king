// 15j_qmma_e3m2_e3m2_f16.cu
//
// Narrow-input FP16 accumulator test. Checks whether QMMA accumulator dtype
// encoding remains the same with FP6 inputs.

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void qmma_e3m2_e3m2_f16_kernel(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* d) {
    const unsigned tid = threadIdx.x;
    uint32_t a0 = a[tid * 4 + 0], a1 = a[tid * 4 + 1], a2 = a[tid * 4 + 2], a3 = a[tid * 4 + 3];
    uint32_t b0 = b[tid * 2 + 0], b1 = b[tid * 2 + 1];
    uint32_t c0 = c[tid * 2 + 0], c1 = c[tid * 2 + 1];
    uint32_t d0, d1;
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e3m2.e3m2.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    d[tid * 2 + 0] = d0;
    d[tid * 2 + 1] = d1;
}

int main(void) {
    const int n_a = 128, n_b = 64, n_c = 64, n_d = 64;
    uint32_t *h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    uint32_t *h_c = (uint32_t*)malloc(n_c * sizeof(uint32_t));
    uint32_t *h_d = (uint32_t*)malloc(n_d * sizeof(uint32_t));
    for (int i = 0; i < n_a; ++i) h_a[i] = 0;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0;
    for (int i = 0; i < n_c; ++i) h_c[i] = 0;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0;
    uint32_t *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t)); cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_c, n_c * sizeof(uint32_t)); cudaMalloc(&d_d, n_d * sizeof(uint32_t));
    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n_c * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, n_d * sizeof(uint32_t), cudaMemcpyHostToDevice);
    qmma_e3m2_e3m2_f16_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, n_d * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("d[0] = 0x%08x\n", h_d[0]);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    free(h_a); free(h_b); free(h_c); free(h_d);
    return 0;
}

