// Sparse block-scaled MMA probe.

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef KIND
#define KIND mxf8f6f4
#endif

#ifndef SHAPE
#define SHAPE m16n8k64
#endif

#ifndef SCALE_VEC
#define SCALE_VEC scale_vec::1X
#endif

#ifndef A_DTYPE
#define A_DTYPE e4m3
#endif

#ifndef B_DTYPE
#define B_DTYPE e4m3
#endif

#ifndef SCALE_DTYPE
#define SCALE_DTYPE ue8m0
#endif

#ifndef SCALE_VALUE
#define SCALE_VALUE 0x7fu
#endif

#ifndef META_VALUE
#define META_VALUE 0xaaaaaaaau
#endif

#ifndef SELECTOR_VALUE
#define SELECTOR_VALUE 0
#endif

#define STR2(x) #x
#define STR(x) STR2(x)

__global__ void sparse_block_scale_probe(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ d)
{
    const unsigned tid = threadIdx.x;

    uint32_t a0 = a[tid * 4 + 0];
    uint32_t a1 = a[tid * 4 + 1];
    uint32_t a2 = a[tid * 4 + 2];
    uint32_t a3 = a[tid * 4 + 3];

    uint32_t b0 = b[tid * 4 + 0];
    uint32_t b1 = b[tid * 4 + 1];
    uint32_t b2 = b[tid * 4 + 2];
    uint32_t b3 = b[tid * 4 + 3];

    float c0 = c[tid * 4 + 0];
    float c1 = c[tid * 4 + 1];
    float c2 = c[tid * 4 + 2];
    float c3 = c[tid * 4 + 3];

    uint32_t meta = META_VALUE;
    uint32_t sfa = SCALE_VALUE;
    uint32_t sfb = SCALE_VALUE;
    uint16_t bid = 0;
    uint16_t tid_sf = 0;
    float d0, d1, d2, d3;

    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned." STR(SHAPE) ".row.col.kind::"
        STR(KIND) ".block_scale." STR(SCALE_VEC) ".f32."
        STR(A_DTYPE) "." STR(B_DTYPE) ".f32." STR(SCALE_DTYPE) " "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, "
        "%16, " STR(SELECTOR_VALUE) ", "
        "{%17}, {%18, %19}, "
        "{%20}, {%21, %22};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(meta),
          "r"(sfa), "h"(bid), "h"(tid_sf),
          "r"(sfb), "h"(bid), "h"(tid_sf)
    );

    d[tid * 4 + 0] = d0;
    d[tid * 4 + 1] = d1;
    d[tid * 4 + 2] = d2;
    d[tid * 4 + 3] = d3;
}

int main(void)
{
    const int n_a = 128;
    const int n_b = 128;
    const int n_c = 128;
    const int n_d = 128;

    uint32_t* h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t* h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    float* h_c = (float*)malloc(n_c * sizeof(float));
    float* h_d = (float*)malloc(n_d * sizeof(float));

    for (int i = 0; i < n_a; ++i) h_a[i] = 0x38383838u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x38383838u;
    for (int i = 0; i < n_c; ++i) h_c[i] = 0.0f;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0.0f;

    uint32_t *d_a, *d_b;
    float *d_c, *d_d;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t));
    cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_c, n_c * sizeof(float));
    cudaMalloc(&d_d, n_d * sizeof(float));

    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n_c * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, n_d * sizeof(float), cudaMemcpyHostToDevice);

    sparse_block_scale_probe<<<1, 32>>>(d_a, d_b, d_c, d_d);
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, n_d * sizeof(float), cudaMemcpyDeviceToHost);

    printf("d[0] = %f\n", h_d[0]);

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
