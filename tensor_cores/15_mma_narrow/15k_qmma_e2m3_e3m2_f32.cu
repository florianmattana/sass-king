// 15k_qmma_e2m3_e3m2_f32.cu
//
// Operand-order asymmetry test. Reverses 15c: A is E2M3, B is E3M2.

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void qmma_e2m3_e3m2_f32_kernel(const uint32_t* a, const uint32_t* b, const float* c, float* d) {
    const unsigned tid = threadIdx.x;
    uint32_t a0 = a[tid * 4 + 0], a1 = a[tid * 4 + 1], a2 = a[tid * 4 + 2], a3 = a[tid * 4 + 3];
    uint32_t b0 = b[tid * 2 + 0], b1 = b[tid * 2 + 1];
    float c0 = c[tid * 4 + 0], c1 = c[tid * 4 + 1], c2 = c[tid * 4 + 2], c3 = c[tid * 4 + 3];
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e3m2.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    d[tid * 4 + 0] = d0;
    d[tid * 4 + 1] = d1;
    d[tid * 4 + 2] = d2;
    d[tid * 4 + 3] = d3;
}

int main(void) {
    const int n_a = 128, n_b = 64, n_c = 128, n_d = 128;
    uint32_t *h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    float *h_c = (float*)malloc(n_c * sizeof(float));
    float *h_d = (float*)malloc(n_d * sizeof(float));
    for (int i = 0; i < n_a; ++i) h_a[i] = 0;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0;
    for (int i = 0; i < n_c; ++i) h_c[i] = 0.0f;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0.0f;
    uint32_t *d_a, *d_b; float *d_c, *d_d;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t)); cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_c, n_c * sizeof(float)); cudaMalloc(&d_d, n_d * sizeof(float));
    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n_c * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, n_d * sizeof(float), cudaMemcpyHostToDevice);
    qmma_e2m3_e3m2_f32_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);
    printf("d[0] = %f\n", h_d[0]);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    free(h_a); free(h_b); free(h_c); free(h_d);
    return 0;
}
