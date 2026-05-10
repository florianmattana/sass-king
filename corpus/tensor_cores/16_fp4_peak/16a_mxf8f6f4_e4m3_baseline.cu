// 16a_mxf8f6f4_e4m3_baseline.cu
//
// Chapter 16 variant a. Block-scaled MMA: kind::mxf8f6f4 with E4M3 inputs
// and ue8m0 scale factors. Baseline for the block-scaled FP8/FP6/FP4 family.
//
// Compared to chapter 14a (same shape m16n8k32, same E4M3.E4M3, same F32 acc),
// this adds block_scale and scale_vec::1X to the PTX mnemonic, plus scale factor
// operands for A and B. Scale factors are ue8m0 (unsigned 8 exp bits, 0 mantissa).
//
// Scale factor = 1.0 is ue8m0 bit pattern 0x7F (exponent = 127 bias).
// With VS=32 and all scales = 1.0, the output should match kernel 14a.
//
// Expected d[0] = 32.0 (identical to 14a: 32 products of 1.0 * 1.0).
//
// Matches CUTLASS SM120_16x8x32_TN_VS<float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 32>
// from mma_sm120.hpp (line 2763).
//
// Compile: nvcc -arch=compute_120a -code=sm_120a -o 16a_mxf8f6f4_e4m3_baseline 16a_mxf8f6f4_e4m3_baseline.cu
// Dump:    cuobjdump --dump-sass 16a_mxf8f6f4_e4m3_baseline > 16a_mxf8f6f4_e4m3_baseline.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void mxf8f6f4_e4m3_kernel(
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

    // Scale factors: ue8m0 with exp = 127 (bias) = 2^0 = 1.0
    // Stored as uint8 = 0x7F, cast to uint32 for PTX
    uint32_t sfa = 0x7Fu;
    uint32_t sfb = 0x7Fu;
    uint16_t bid = 0;
    uint16_t tid_sf = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, "
        "{%17}, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),
           "r"(b0), "r"(b1),
           "f"(c0), "f"(c1), "f"(c2), "f"(c3),
           "r"(sfa), "h"(bid), "h"(tid_sf),
           "r"(sfb), "h"(bid), "h"(tid_sf)
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

    // E4M3 1.0 = 0x38, packed 0x38383838
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x38383838u;
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

    mxf8f6f4_e4m3_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    // Scale factor = 1.0, so result should match kernel 14a: 32 products of 1.0 * 1.0
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
