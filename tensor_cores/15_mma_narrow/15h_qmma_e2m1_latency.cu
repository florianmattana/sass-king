// 15h_qmma_e2m1_latency.cu
//
// Serial latency microbenchmark for unscaled FP4 E2M1 QMMA at m16n8k32.

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef N_QMMA
#define N_QMMA 16
#endif

#define ONE_QMMA \
    "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 " \
    "{%0, %1, %2, %3}, " \
    "{%4, %5, %6, %7}, " \
    "{%8, %9}, " \
    "{%0, %1, %2, %3};\n"

#define QMMA_2 ONE_QMMA ONE_QMMA
#define QMMA_4 QMMA_2 QMMA_2
#define QMMA_8 QMMA_4 QMMA_4
#define QMMA_16 QMMA_8 QMMA_8
#define QMMA_32 QMMA_16 QMMA_16
#define QMMA_64 QMMA_32 QMMA_32

#if N_QMMA == 16
#define QMMA_CHAIN QMMA_16
#elif N_QMMA == 32
#define QMMA_CHAIN QMMA_32
#elif N_QMMA == 64
#define QMMA_CHAIN QMMA_64
#else
#error "N_QMMA must be 16, 32, or 64"
#endif

__global__ void qmma_latency_kernel(const uint32_t* a, const uint32_t* b, const float* c, float* d, unsigned long long* cycles_out) {
    const unsigned tid = threadIdx.x;
    uint32_t a0 = a[tid * 4 + 0], a1 = a[tid * 4 + 1], a2 = a[tid * 4 + 2], a3 = a[tid * 4 + 3];
    uint32_t b0 = b[tid * 2 + 0], b1 = b[tid * 2 + 1];
    float c0 = c[tid * 4 + 0], c1 = c[tid * 4 + 1], c2 = c[tid * 4 + 2], c3 = c[tid * 4 + 3];
    unsigned long long start = clock64();
    asm volatile(QMMA_CHAIN
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    unsigned long long end = clock64();
    d[tid * 4 + 0] = c0; d[tid * 4 + 1] = c1; d[tid * 4 + 2] = c2; d[tid * 4 + 3] = c3;
    if (tid == 0) *cycles_out = end - start;
}

int main(void) {
    const int n_a = 128, n_b = 64, n_c = 128, n_d = 128;
    uint32_t *h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    float *h_c = (float*)malloc(n_c * sizeof(float));
    float *h_d = (float*)malloc(n_d * sizeof(float));
    unsigned long long h_cycles = 0;
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x22222222u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x22222222u;
    for (int i = 0; i < n_c; ++i) h_c[i] = 0.0f;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0.0f;
    uint32_t *d_a, *d_b; float *d_c, *d_d; unsigned long long *d_cycles;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t)); cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_c, n_c * sizeof(float)); cudaMalloc(&d_d, n_d * sizeof(float));
    cudaMalloc(&d_cycles, sizeof(unsigned long long));
    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n_c * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, n_d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cycles, &h_cycles, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    qmma_latency_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d, d_cycles);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);
    printf("N=%d total_cycles=%llu cycles_per_qmma=%.2f d[0]=%.1f\n", N_QMMA, h_cycles, (double)h_cycles / N_QMMA, h_d[0]);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d); cudaFree(d_cycles);
    free(h_a); free(h_b); free(h_c); free(h_d);
    return 0;
}

