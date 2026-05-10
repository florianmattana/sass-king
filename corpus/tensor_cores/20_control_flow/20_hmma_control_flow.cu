// 20_hmma_control_flow.cu
//
// Chapter 20 HMMA control-flow probes.
//
// Compile examples:
//   nvcc -arch=sm_120 -DVARIANT=206 -o build/20g_nested_hmma_4x2 20_hmma_control_flow.cu
//   cuobjdump --dump-sass build/20g_nested_hmma_4x2 > 20g_nested_hmma_4x2.sass

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef VARIANT
#define VARIANT 206
#endif

__device__ __forceinline__ void hmma_step(const uint32_t a0,
                                          const uint32_t a1,
                                          const uint32_t a2,
                                          const uint32_t a3,
                                          const uint32_t b0,
                                          const uint32_t b1,
                                          float& d0,
                                          float& d1,
                                          float& d2,
                                          float& d3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
}

__global__ void hmma_control_flow_kernel(const uint32_t* __restrict__ a,
                                         const uint32_t* __restrict__ b,
                                         float* __restrict__ out) {
    const unsigned tid = threadIdx.x;

    const uint32_t a0 = a[tid * 4 + 0];
    const uint32_t a1 = a[tid * 4 + 1];
    const uint32_t a2 = a[tid * 4 + 2];
    const uint32_t a3 = a[tid * 4 + 3];
    const uint32_t b0 = b[tid * 2 + 0];
    const uint32_t b1 = b[tid * 2 + 1];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

#if VARIANT == 206
    // 20g: nested constant loops, 4 x 2 HMMA body.
    for (int o = 0; o < 4; ++o) {
        for (int k = 0; k < 2; ++k) {
            hmma_step(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
        }
    }
#elif VARIANT == 207
    // 20h: nested constant loops, 8 x 2 HMMA body.
    for (int o = 0; o < 8; ++o) {
        for (int k = 0; k < 2; ++k) {
            hmma_step(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
        }
    }
#else
#error "Unsupported VARIANT for hmma_control_flow_kernel"
#endif

    out[tid * 4 + 0] = d0;
    out[tid * 4 + 1] = d1;
    out[tid * 4 + 2] = d2;
    out[tid * 4 + 3] = d3;
}

int main() {
    constexpr int n_a = 128;
    constexpr int n_b = 64;
    constexpr int n_out = 128;

    uint32_t* h_a = static_cast<uint32_t*>(malloc(n_a * sizeof(uint32_t)));
    uint32_t* h_b = static_cast<uint32_t*>(malloc(n_b * sizeof(uint32_t)));
    float* h_out = static_cast<float*>(malloc(n_out * sizeof(float)));
    for (int i = 0; i < n_a; ++i) h_a[i] = 0x3c003c00u;
    for (int i = 0; i < n_b; ++i) h_b[i] = 0x3c003c00u;
    for (int i = 0; i < n_out; ++i) h_out[i] = 0.0f;

    uint32_t *d_a, *d_b;
    float* d_out;
    cudaMalloc(&d_a, n_a * sizeof(uint32_t));
    cudaMalloc(&d_b, n_b * sizeof(uint32_t));
    cudaMalloc(&d_out, n_out * sizeof(float));
    cudaMemcpy(d_a, h_a, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);

    hmma_control_flow_kernel<<<1, 32>>>(d_a, d_b, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost);
    printf("out[0] = %f\n", h_out[0]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_a);
    free(h_b);
    free(h_out);
    return 0;
}
