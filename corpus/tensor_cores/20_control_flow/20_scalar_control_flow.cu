// 20_scalar_control_flow.cu
//
// Chapter 20 scalar control-flow probes.
//
// Compile examples:
//   nvcc -arch=sm_120 -DVARIANT=200 -o build/20a_constant_loop_n4 20_scalar_control_flow.cu
//   cuobjdump --dump-sass build/20a_constant_loop_n4 > 20a_constant_loop_n4.sass

#include <cstdio>
#include <cstdlib>

#ifndef VARIANT
#define VARIANT 200
#endif

#define NOINLINE __attribute__((noinline))

__device__ __forceinline__ float step_dep(float acc, float x, int i) {
    return acc * 1.001f + x + static_cast<float>(i);
}

template <int OUTER, int INNER>
__global__ void template_nested_kernel(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float acc = in[tid];
    #pragma unroll
    for (int o = 0; o < OUTER; ++o) {
        #pragma unroll
        for (int k = 0; k < INNER; ++k) {
            acc = step_dep(acc, in[tid], o * INNER + k);
        }
    }
    out[tid] = acc;
}

template <int OUTER, int INNER>
__global__ void template_nested_unique_store_kernel(const float* __restrict__ in,
                                                    float* __restrict__ out,
                                                    int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float acc = in[tid];
    #pragma unroll
    for (int o = 0; o < OUTER; ++o) {
        #pragma unroll
        for (int k = 0; k < INNER; ++k) {
            acc = step_dep(acc, in[tid], o * INNER + k);
            out[(o * INNER + k) * n + tid] = acc;
        }
    }
}

template <int OUTER, int INNER>
__global__ void template_nested_identical_kernel(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float acc = 0.0f;
    const float x = in[tid];
    #pragma unroll
    for (int o = 0; o < OUTER; ++o) {
        #pragma unroll
        for (int k = 0; k < INNER; ++k) {
            acc += x;
        }
    }
    out[tid] = acc;
}

#if VARIANT <= 217
__global__ void scalar_control_flow_kernel(const float* __restrict__ in,
                                           float* __restrict__ out,
                                           int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float acc = in[tid];

#if VARIANT == 200
    // 20a: constant loop N=4, default unroll policy.
    for (int i = 0; i < 4; ++i) {
        acc = step_dep(acc, in[tid], i);
    }
    out[tid] = acc;

#elif VARIANT == 201
    // 20b: constant loop N=16, default unroll policy.
    for (int i = 0; i < 16; ++i) {
        acc = step_dep(acc, in[tid], i);
    }
    out[tid] = acc;

#elif VARIANT == 202
    // 20c: dynamic loop, trip count from kernel argument.
    for (int i = 0; i < n; ++i) {
        acc = step_dep(acc, in[tid], i);
    }
    out[tid] = acc;

#elif VARIANT == 203
    // 20d: constant loop N=16, pragma prevents unrolling.
    #pragma unroll 1
    for (int i = 0; i < 16; ++i) {
        acc = step_dep(acc, in[tid], i);
    }
    out[tid] = acc;

#elif VARIANT == 204
    // 20e: nested constant loops, 4 x 2 scalar body.
    for (int o = 0; o < 4; ++o) {
        for (int k = 0; k < 2; ++k) {
            acc = step_dep(acc, in[tid], o * 2 + k);
        }
    }
    out[tid] = acc;

#elif VARIANT == 205
    // 20f: nested constant loops, 8 x 2 scalar body.
    for (int o = 0; o < 8; ++o) {
        for (int k = 0; k < 2; ++k) {
            acc = step_dep(acc, in[tid], o * 2 + k);
        }
    }
    out[tid] = acc;

#elif VARIANT == 208
    // 20i: dynamic loop with short predicated body.
    for (int i = 0; i < n; ++i) {
        if ((i & 1) == 0) {
            acc += in[tid];
        }
    }
    out[tid] = acc;

#elif VARIANT == 209
    // 20j: dynamic loop with larger conditional body.
    for (int i = 0; i < n; ++i) {
        if ((i & 1) == 0) {
            acc = step_dep(acc, in[tid], i);
            acc = step_dep(acc, in[tid], i + 1);
            acc = step_dep(acc, in[tid], i + 2);
            acc = step_dep(acc, in[tid], i + 3);
        } else {
            acc += 0.5f;
        }
    }
    out[tid] = acc;

#elif VARIANT == 210
    // 20k: dynamic loop with break.
    for (int i = 0; i < n; ++i) {
        acc = step_dep(acc, in[tid], i);
        if (acc > 2048.0f) break;
    }
    out[tid] = acc;

#elif VARIANT == 211
    // 20l: dynamic loop with continue.
    for (int i = 0; i < n; ++i) {
        if ((i & 3) == 0) continue;
        acc = step_dep(acc, in[tid], i);
    }
    out[tid] = acc;

#elif VARIANT == 212
    // 20m: constant loop N=16, explicit full unroll.
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        acc = step_dep(acc, in[tid], i);
    }
    out[tid] = acc;

#elif VARIANT == 213
    // 20n: constant loop N=16, explicit partial unroll by 4.
    #pragma unroll 4
    for (int i = 0; i < 16; ++i) {
        acc = step_dep(acc, in[tid], i);
    }
    out[tid] = acc;

#elif VARIANT == 214
    // 20o: dynamic loop with explicit partial unroll by 4.
    #pragma unroll 4
    for (int i = 0; i < n; ++i) {
        acc = step_dep(acc, in[tid], i);
    }
    out[tid] = acc;

#elif VARIANT == 215
    // 20p: dynamic loop with loop-carried accumulator dependency.
    for (int i = 0; i < n; ++i) {
        acc = acc * 1.001f + in[tid];
    }
    out[tid] = acc;

#elif VARIANT == 216
    // 20q: dynamic loop with four independent accumulators.
    float a0 = acc;
    float a1 = acc + 1.0f;
    float a2 = acc + 2.0f;
    float a3 = acc + 3.0f;
    for (int i = 0; i < n; ++i) {
        const float x = in[tid];
        a0 = a0 * 1.001f + x;
        a1 = a1 * 1.002f + x;
        a2 = a2 * 1.003f + x;
        a3 = a3 * 1.004f + x;
    }
    out[tid] = a0 + a1 + a2 + a3;

#elif VARIANT == 217
    // 20r: dynamic loop with volatile side-effect store.
    volatile float* vout = out;
    for (int i = 0; i < n; ++i) {
        acc = step_dep(acc, in[tid], i);
        vout[(i & 3) * n + tid] = acc;
    }

#else
#error "Unsupported VARIANT for scalar_control_flow_kernel"
#endif
}
#endif

int main() {
    constexpr int n = 32;
    constexpr int out_mul =
#if VARIANT == 217
        4;
#elif VARIANT == 220
        16;
#else
        1;
#endif

    float* h_in = static_cast<float*>(malloc(n * sizeof(float)));
    float* h_out = static_cast<float*>(malloc(n * out_mul * sizeof(float)));
    for (int i = 0; i < n; ++i) h_in[i] = 1.0f + static_cast<float>(i & 3);
    for (int i = 0; i < n * out_mul; ++i) h_out[i] = 0.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * out_mul * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, n * out_mul * sizeof(float), cudaMemcpyHostToDevice);

#if VARIANT == 218
    template_nested_kernel<8, 2><<<1, 32>>>(d_in, d_out, n);
#elif VARIANT == 219
    template_nested_kernel<8, 2><<<1, 32>>>(d_in, d_out, n);
    template_nested_kernel<4, 2><<<1, 32>>>(d_in, d_out, n);
#elif VARIANT == 220
    template_nested_unique_store_kernel<8, 2><<<1, 32>>>(d_in, d_out, n);
#elif VARIANT == 221
    template_nested_identical_kernel<8, 2><<<1, 32>>>(d_in, d_out, n);
#else
    scalar_control_flow_kernel<<<1, 32>>>(d_in, d_out, n);
#endif

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, n * out_mul * sizeof(float), cudaMemcpyDeviceToHost);
    printf("out[0] = %f\n", h_out[0]);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
