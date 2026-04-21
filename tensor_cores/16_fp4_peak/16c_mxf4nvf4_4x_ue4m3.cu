// 16c_mxf4nvf4_4x_ue4m3.cu
//
// Chapter 16 variant c. Block-scaled MMA with kind::mxf4nvf4, scale_vec::4X,
// shape m16n8k64, e2m1 FP4 inputs, ue4m3 scales.
// This is the FP4 peak path (900+ TFLOPS announced by NVIDIA for consumer Blackwell).
//
// Compared to 16b (mxf4nvf4 scale_vec::2X with ue8m0):
//   - scale_vec::4X instead of 2X (4 scales per fragment instead of 2)
//   - ue4m3 scales instead of ue8m0 (4-bit-exp + 3-bit-mantissa instead of 8-bit-exp)
//
// ue4m3 = unsigned E4M3: exp_bias = 7, 1.0 encoded as 0x38 (same as E4M3 sign-free).
// 4 scales packed = 4 bytes = 0x38383838.
//
// With VS=32, each scale covers 32 k-elements. With m16n8k64, 4 scales * 16 = 64.
// Exact semantics of 4X encoding TBD empirically.
//
// Compile: nvcc -arch=compute_120a -code=sm_120a -o 16c_mxf4nvf4_4x_ue4m3 16c_mxf4nvf4_4x_ue4m3.cu
// Dump:    cuobjdump --dump-sass 16c_mxf4nvf4_4x_ue4m3 > 16c_mxf4nvf4_4x_ue4m3.sass

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void mxf4nvf4_4x_ue4m3_kernel(
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

    // 4 scales packed in uint32: ue4m3 = 1.0 is 0x38, four packed = 0x38383838
    uint32_t sfa = 0x38383838u;
    uint32_t sfb = 0x38383838u;
    uint16_t bid = 0;
    uint16_t tid_sf = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
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

    mxf4nvf4_4x_ue4m3_kernel<<<1, 32>>>(d_a, d_b, d_c, d_d);
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
