// 21_divergence_reconvergence.cu
//
// Chapter 21 divergence/reconvergence probes.
//
// Compile examples:
//   nvcc -arch=sm_120 -DVARIANT=2100 -o build/21a_predicated_if 21_divergence_reconvergence.cu
//   cuobjdump --dump-sass build/21a_predicated_if > 21a_predicated_if.sass

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef VARIANT
#define VARIANT 2100
#endif

__device__ __forceinline__ float step_dep(float acc, float x, int i) {
    return acc * 1.001f + x + static_cast<float>(i);
}

__device__ __noinline__ float heavy_call(float x, int lane) {
    float acc = x;
    #pragma unroll 1
    for (int i = 0; i < 8; ++i) {
        acc = acc * 1.003f + static_cast<float>(lane + i);
    }
    return acc;
}

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

__global__ void divergence_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  const uint32_t* __restrict__ a,
                                  const uint32_t* __restrict__ b,
                                  int n) {
    __shared__ float smem[256];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local = threadIdx.x;
    const int lane = local & 31;
    if (tid >= n) return;

    float x = in[tid];
    float acc = x;

#if VARIANT == 2100
    // 21a: simple lane-dependent if, small body.
    if (lane & 1) {
        acc += 1.0f;
    }
    out[tid] = acc;

#elif VARIANT == 2101
    // 21b: uniform branch condition across the warp.
    if (blockIdx.x & 1) {
        acc = step_dep(acc, x, 1);
    } else {
        acc = step_dep(acc, x, 2);
    }
    out[tid] = acc;

#elif VARIANT == 2102
    // 21c: lane-divergent if with a medium arithmetic body.
    if (lane & 1) {
        acc = step_dep(acc, x, lane);
        acc = step_dep(acc, x, lane + 1);
        acc = step_dep(acc, x, lane + 2);
    }
    out[tid] = acc;

#elif VARIANT == 2103
    // 21d: lane-divergent if/else with work on both paths.
    if (lane & 1) {
        acc = step_dep(acc, x, 3);
        acc = step_dep(acc, x, 4);
    } else {
        acc = acc - x * 0.5f;
        acc = acc + 7.0f;
    }
    out[tid] = acc;

#elif VARIANT == 2104
    // 21e: nested lane-dependent divergence.
    if (lane & 1) {
        if (lane & 2) {
            acc = step_dep(acc, x, 5);
        } else {
            acc += 2.0f;
        }
    } else {
        if (lane & 4) {
            acc = step_dep(acc, x, 6);
        } else {
            acc -= 3.0f;
        }
    }
    out[tid] = acc;

#elif VARIANT == 2105
    // 21f: loop with lane-dependent break.
    #pragma unroll 1
    for (int i = 0; i < 8; ++i) {
        acc = step_dep(acc, x, i);
        if (i == (lane & 7)) break;
    }
    out[tid] = acc;

#elif VARIANT == 2106
    // 21g: loop with lane-dependent continue.
    #pragma unroll 1
    for (int i = 0; i < 8; ++i) {
        if (i == (lane & 7)) continue;
        acc = step_dep(acc, x, i);
    }
    out[tid] = acc;

#elif VARIANT == 2107
    // 21h: lane-dependent early return.
    if (lane & 1) return;
    out[tid] = acc + 1.0f;

#elif VARIANT == 2108
    // 21i: divergent arithmetic followed by a block barrier.
    smem[local] = x;
    if (lane & 1) {
        smem[local] = x + 1.0f;
    } else {
        smem[local] = x - 1.0f;
    }
    __syncthreads();
    out[tid] = smem[(local + 1) & 255];

#elif VARIANT == 2109
    // 21j: warp vote around a lane-dependent predicate.
    const bool pred = (x > 0.0f) && ((lane & 1) != 0);
    const unsigned mask = __ballot_sync(0xffffffffu, pred);
    if (lane == 0) {
        reinterpret_cast<unsigned*>(out)[blockIdx.x] = mask;
    }

#elif VARIANT == 2110
    // 21k: select vs branch vs arithmetic mask in one controlled body.
    const float selected = (lane & 1) ? (x + 1.0f) : (x - 1.0f);
    float branched = x;
    if (lane & 2) {
        branched += 2.0f;
    }
    const float masked = x + static_cast<float>(lane & 4);
    out[tid] = selected + branched + masked;

#elif VARIANT == 2111
    // 21l: short divergent body followed by long divergent body.
    if (lane & 1) {
        acc += 1.0f;
    }
    if (lane & 2) {
        acc = step_dep(acc, x, 0);
        acc = step_dep(acc, x, 1);
        acc = step_dep(acc, x, 2);
        acc = step_dep(acc, x, 3);
        acc = step_dep(acc, x, 4);
        acc = step_dep(acc, x, 5);
    }
    out[tid] = acc;

#elif VARIANT == 2112
    // 21m: lane-divergent memory paths.
    if (lane & 1) {
        acc = in[(tid + 32) & (n - 1)];
        out[tid] = acc + 1.0f;
    } else {
        acc = in[(tid + 64) & (n - 1)];
        out[tid] = acc - 1.0f;
    }

#elif VARIANT == 2113
    // 21n: predicate-guarded HMMA region.
    const uint32_t a0 = a[lane * 4 + 0];
    const uint32_t a1 = a[lane * 4 + 1];
    const uint32_t a2 = a[lane * 4 + 2];
    const uint32_t a3 = a[lane * 4 + 3];
    const uint32_t b0 = b[lane * 2 + 0];
    const uint32_t b1 = b[lane * 2 + 1];
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    if (lane & 1) {
        hmma_step(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
    out[tid] = d0 + d1 + d2 + d3;

#elif VARIANT == 2114
    // 21o: loop trip count depends on lane.
    const int trip = (lane & 7) + 1;
    #pragma unroll 1
    for (int i = 0; i < trip; ++i) {
        acc = step_dep(acc, x, i);
    }
    out[tid] = acc;

#elif VARIANT == 2115
    // 21p: production-style epilogue bounds check.
    const int m = n - (blockIdx.x & 15);
    if (tid < m) {
        out[tid] = acc + 1.0f;
    }

#elif VARIANT == 2116
    // 21q: partial-tile masked writeback.
    const bool store0 = lane < (n & 31);
    const bool store1 = lane < ((n + 7) & 31);
    if (store0) out[tid] = acc;
    if (store1) out[n + tid] = acc + 1.0f;

#elif VARIANT == 2117
    // 21r: divergent call body with register/live-range pressure.
    if (lane & 1) {
        acc = heavy_call(acc, lane);
    } else {
        acc = step_dep(acc, x, lane);
    }
    out[tid] = acc;

#elif VARIANT == 2118
    // 21s: cold error/trap path.
    if ((lane == 3) && (x < -1000000.0f)) {
        asm volatile("trap;\n");
    }
    out[tid] = acc + 1.0f;

#elif VARIANT == 2119
    // 21t: use a warp vote to make branch control uniform.
    const bool pred = (x > 0.0f) && ((lane & 1) != 0);
    const int any = __any_sync(0xffffffffu, pred);
    if (any) {
        acc += __shfl_sync(0xffffffffu, x, 0);
    }
    out[tid] = acc;

#else
#error "Unsupported VARIANT"
#endif
}

int main() {
    constexpr int n = 256;
    constexpr int out_n = 512;
    constexpr int a_n = 128;
    constexpr int b_n = 64;

    float* h_in = static_cast<float*>(malloc(n * sizeof(float)));
    float* h_out = static_cast<float*>(malloc(out_n * sizeof(float)));
    uint32_t* h_a = static_cast<uint32_t*>(malloc(a_n * sizeof(uint32_t)));
    uint32_t* h_b = static_cast<uint32_t*>(malloc(b_n * sizeof(uint32_t)));

    for (int i = 0; i < n; ++i) h_in[i] = 1.0f + static_cast<float>(i & 7);
    for (int i = 0; i < out_n; ++i) h_out[i] = 0.0f;
    for (int i = 0; i < a_n; ++i) h_a[i] = 0x3c003c00u;
    for (int i = 0; i < b_n; ++i) h_b[i] = 0x3c003c00u;

    float *d_in, *d_out;
    uint32_t *d_a, *d_b;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, out_n * sizeof(float));
    cudaMalloc(&d_a, a_n * sizeof(uint32_t));
    cudaMalloc(&d_b, b_n * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, out_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, a_n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    divergence_kernel<<<1, 256>>>(d_in, d_out, d_a, d_b, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("out[0] = %f\n", h_out[0]);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_in);
    free(h_out);
    free(h_a);
    free(h_b);
    return 0;
}
