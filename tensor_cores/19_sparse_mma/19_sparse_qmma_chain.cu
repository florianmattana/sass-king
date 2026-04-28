// Sparse kind::f8f6f4 MMA dependency-chain probe.

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef N_MMA
#define N_MMA 16
#endif

#ifndef A_DTYPE
#define A_DTYPE e4m3
#endif

#ifndef B_DTYPE
#define B_DTYPE e4m3
#endif

#define STR2(x) #x
#define STR(x) STR2(x)

__global__ void sparse_qmma_chain(
    const uint32_t* __restrict__ a,
    const uint32_t* __restrict__ b,
    uint64_t* __restrict__ cycles,
    float* __restrict__ out)
{
    uint32_t a0 = a[0];
    uint32_t a1 = a[1];
    uint32_t a2 = a[2];
    uint32_t a3 = a[3];
    uint32_t b0 = b[0];
    uint32_t b1 = b[1];
    uint32_t b2 = b[2];
    uint32_t b3 = b[3];
    uint32_t meta = 0xaaaaaaaau;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    uint64_t start = clock64();
#pragma unroll
    for (int i = 0; i < N_MMA; ++i) {
        asm volatile(
            "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32."
            STR(A_DTYPE) "." STR(B_DTYPE) ".f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9, %10, %11}, "
            "{%0, %1, %2, %3}, "
            "%12, 0;\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(b2), "r"(b3),
              "r"(meta)
        );
    }
    uint64_t stop = clock64();

    if (threadIdx.x == 0) cycles[0] = stop - start;
    out[threadIdx.x * 4 + 0] = d0;
    out[threadIdx.x * 4 + 1] = d1;
    out[threadIdx.x * 4 + 2] = d2;
    out[threadIdx.x * 4 + 3] = d3;
}

int main(void)
{
    uint32_t h_a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
    uint32_t h_b[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
    uint64_t h_cycles = 0;
    float h_out[128] = {};

    uint32_t *d_a, *d_b;
    uint64_t* d_cycles;
    float* d_out;
    cudaMalloc(&d_a, sizeof(h_a));
    cudaMalloc(&d_b, sizeof(h_b));
    cudaMalloc(&d_cycles, sizeof(uint64_t));
    cudaMalloc(&d_out, sizeof(h_out));
    cudaMemcpy(d_a, h_a, sizeof(h_a), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice);

    sparse_qmma_chain<<<1, 32>>>(d_a, d_b, d_cycles, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    printf("cycles=%llu out0=%f\n", (unsigned long long)h_cycles, h_out[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_cycles);
    cudaFree(d_out);
    return 0;
}
