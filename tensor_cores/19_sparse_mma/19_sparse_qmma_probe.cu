// Sparse kind::f8f6f4 MMA probe.
//
// Compile example:
// nvcc -arch=compute_120a -code=sm_120a -DA_DTYPE=e4m3 -DB_DTYPE=e4m3 \
//   -DACC_F32=1 -DMETA_VALUE=0xaaaaaaaau -DSELECTOR_VALUE=0 \
//   -o 19a_sparse_e4m3_e4m3 19_sparse_qmma_probe.cu

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef A_DTYPE
#define A_DTYPE e4m3
#endif

#ifndef B_DTYPE
#define B_DTYPE e4m3
#endif

#ifndef ACC_F32
#define ACC_F32 1
#endif

#ifndef META_VALUE
#define META_VALUE 0xaaaaaaaau
#endif

#ifndef SELECTOR_VALUE
#define SELECTOR_VALUE 0
#endif

#define STR2(x) #x
#define STR(x) STR2(x)

#if ACC_F32
__global__ void sparse_qmma_probe(
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
    float d0, d1, d2, d3;

    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32."
        STR(A_DTYPE) "." STR(B_DTYPE) ".f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, "
        "%16, " STR(SELECTOR_VALUE) ";\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(meta)
    );

    d[tid * 4 + 0] = d0;
    d[tid * 4 + 1] = d1;
    d[tid * 4 + 2] = d2;
    d[tid * 4 + 3] = d3;
}
#else
__global__ void sparse_qmma_probe(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    const uint32_t* __restrict__ c,
    uint32_t* __restrict__ d)
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

    uint32_t c0 = c[tid * 2 + 0];
    uint32_t c1 = c[tid * 2 + 1];

    uint32_t meta = META_VALUE;
    uint32_t d0, d1;

    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f16."
        STR(A_DTYPE) "." STR(B_DTYPE) ".f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7, %8, %9}, "
        "{%10, %11}, "
        "%12, " STR(SELECTOR_VALUE) ";\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "r"(c0), "r"(c1),
          "r"(meta)
    );

    d[tid * 2 + 0] = d0;
    d[tid * 2 + 1] = d1;
}
#endif

int main(void)
{
    const int n_a = 128;
    const int n_b = 128;
    const int n_c = ACC_F32 ? 128 : 64;
    const int n_d = ACC_F32 ? 128 : 64;

    uint32_t* h_a = (uint32_t*)malloc(n_a * sizeof(uint32_t));
    uint32_t* h_b = (uint32_t*)malloc(n_b * sizeof(uint32_t));
    uint32_t* h_c = (uint32_t*)malloc(n_c * sizeof(uint32_t));
    uint32_t* h_d = (uint32_t*)malloc(n_d * sizeof(uint32_t));

    for (int i = 0; i < n_a; ++i) h_a[i] = 0x38383838u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x38383838u;
    for (int i = 0; i < n_c; ++i) h_c[i] = 0;
    for (int i = 0; i < n_d; ++i) h_d[i] = 0;

    uint32_t *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t));
    cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_c, n_c * sizeof(uint32_t));
    cudaMalloc(&d_d, n_d * sizeof(uint32_t));

    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, n_c * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, n_d * sizeof(uint32_t), cudaMemcpyHostToDevice);

#if ACC_F32
    sparse_qmma_probe<<<1, 32>>>(d_a, d_b, (const float*)d_c, (float*)d_d);
#else
    sparse_qmma_probe<<<1, 32>>>(d_a, d_b, d_c, d_d);
#endif
    cudaDeviceSynchronize();
    cudaMemcpy(h_d, d_d, n_d * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("d[0] raw = 0x%08x\n", h_d[0]);

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
