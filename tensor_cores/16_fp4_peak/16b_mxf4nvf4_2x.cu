// 16b_mxf4nvf4_2x.cu
//
// Chapter 16 variant b. Block-scaled MMA with kind::mxf4nvf4 (NVidia FP4),
// scale_vec::2X, shape m16n8k64, e2m1 inputs, ue8m0 scales.
//
// Compared to 16a (kind::mxf8f6f4 scale_vec::1X m16n8k32):
//   - New kind: mxf4nvf4 (FP4 specific)
//   - New shape: m16n8k64 (2x k dim)
//   - New scale vec: 2X (2 scale factors per A/B fragment)
//
// With scale_vec::2X, each of A and B has 2 scales, packed into the input uint32.
// ue8m0 scale = 1.0 is bit pattern 0x7F, two packed = 0x7F7F (16 bits).
//
// With all inputs and scales = 1.0, expected output per element:
//   d = sum over k=64 of (1.0 * 1.0 * 1.0 * 1.0) = 64.0
//
// Matches CUTLASS SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 32>
// Note: the VS=32 branch in CUTLASS uses scale_vec::2X.
//
// Compile: nvcc -arch=compute_120a -code=sm_120a -o 16b_mxf4nvf4_2x 16b_mxf4nvf4_2x.cu
// Dump:    cuobjdump --dump-sass 16b_mxf4nvf4_2x > 16b_mxf4nvf4_2x.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void mxf4nvf4_2x_kernel(
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

    // Scale factors: 2X mode with ue8m0 (2 scales per fragment, 8 bits each)
    // Both scales = 1.0 = 0x7F, packed: 0x7F7F (16 bits low of uint32)
    uint32_t sfa = 0x00007F7Fu;
    uint32_t sfb = 0x00007F7Fu;
    uint16_t bid = 0;
    uint16_t tid_sf = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
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

    // Use zero inputs to get deterministic d[0] = 0 (avoids FP4 layout unknown)
    // The SASS observations are independent of input values.
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x00000000u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x00000000u;
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

    mxf4nvf4_2x_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    printf("d[0] = %f (expected 0.0)\n", h_d[0]);

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
