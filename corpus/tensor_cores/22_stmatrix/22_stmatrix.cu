// 22_stmatrix.cu
//
// Chapter 22 stmatrix / matrix-store probes.
//
// Compile examples:
//   nvcc -arch=sm_120 -DVARIANT=2200 -o build/22a_stmatrix_x1 22_stmatrix.cu
//   cuobjdump --dump-sass build/22a_stmatrix_x1 > 22a_stmatrix_x1.sass

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                cudaGetErrorString(status));                                    \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

#ifndef VARIANT
#define VARIANT 2200
#endif

__device__ __forceinline__ uint32_t smem_u32_ptr(void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void stmatrix_x1(uint32_t addr, uint32_t r0) {
    asm volatile(
        "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n"
        :: "r"(addr), "r"(r0)
        : "memory");
}

__device__ __forceinline__ void stmatrix_x2(uint32_t addr, uint32_t r0, uint32_t r1) {
    asm volatile(
        "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
        :: "r"(addr), "r"(r0), "r"(r1)
        : "memory");
}

__device__ __forceinline__ void stmatrix_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
        : "memory");
}

__device__ __forceinline__ void stmatrix_x1_trans(uint32_t addr, uint32_t r0) {
    asm volatile(
        "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n"
        :: "r"(addr), "r"(r0)
        : "memory");
}

__device__ __forceinline__ void stmatrix_x2_trans(uint32_t addr, uint32_t r0, uint32_t r1) {
    asm volatile(
        "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n"
        :: "r"(addr), "r"(r0), "r"(r1)
        : "memory");
}

__device__ __forceinline__ void stmatrix_x4_trans(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile(
        "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
        : "memory");
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

__global__ void stmatrix_kernel(const uint32_t* __restrict__ in,
                                uint32_t* __restrict__ out) {
    __shared__ __align__(16) uint32_t smem[512];

    const int tid = threadIdx.x;
    const uint32_t r0 = in[tid + 0];
    const uint32_t r1 = in[tid + 32];
    const uint32_t r2 = in[tid + 64];
    const uint32_t r3 = in[tid + 96];
    uint32_t addr = smem_u32_ptr(&smem[tid * 4]);

#if VARIANT == 2200
    stmatrix_x1(addr, r0);
#elif VARIANT == 2201
    stmatrix_x2(addr, r0, r1);
#elif VARIANT == 2202
    stmatrix_x4(addr, r0, r1, r2, r3);
#elif VARIANT == 2203
    stmatrix_x1_trans(addr, r0);
#elif VARIANT == 2204
    stmatrix_x2_trans(addr, r0, r1);
#elif VARIANT == 2205
    stmatrix_x4_trans(addr, r0, r1, r2, r3);
#elif VARIANT == 2206
    smem[tid * 4 + 0] = r0;
    smem[tid * 4 + 1] = r1;
    smem[tid * 4 + 2] = r2;
    smem[tid * 4 + 3] = r3;
#elif VARIANT == 2207
    stmatrix_x4(addr, r0, r1, r2, r3);
#elif VARIANT == 2208
    stmatrix_x4_trans(addr, r0, r1, r2, r3);
#elif VARIANT == 2209
    stmatrix_x4(addr, r0, r1, r2, r3);
#elif VARIANT == 2210
    stmatrix_x4(addr, r0, r1, r2, r3);
#elif VARIANT == 2211
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    hmma_step(r0, r1, r2, r3, in[tid + 128], in[tid + 160], d0, d1, d2, d3);
    stmatrix_x4(addr, __float_as_uint(d0), __float_as_uint(d1), __float_as_uint(d2), __float_as_uint(d3));
#else
#error "Unknown VARIANT"
#endif

#if VARIANT == 2210
    out[tid] = smem[tid * 4];
#else
    __syncthreads();

#if VARIANT == 2207 || VARIANT == 2208
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[tid * 4 + i] = smem[tid * 4 + i];
    }
#elif VARIANT == 2209
    out[tid] = smem[((tid + 1) & 31) * 4];
#else
    out[tid] = smem[tid * 4];
#endif
#endif
}

#ifdef TEST_UNSUPPORTED_STMATRIX_B8
__global__ void unsupported_stmatrix_b8_kernel(uint32_t* out) {
    __shared__ __align__(16) uint32_t smem[128];
    const int tid = threadIdx.x;
    const uint32_t addr = smem_u32_ptr(&smem[tid * 4]);
    const uint32_t r0 = static_cast<uint32_t>(tid);
    const uint32_t r1 = r0 + 32;
    const uint32_t r2 = r0 + 64;
    const uint32_t r3 = r0 + 96;

    asm volatile(
        "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
        :: "r"(addr), "r"(r0)
        : "memory");
    asm volatile(
        "stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
        :: "r"(addr), "r"(r0), "r"(r1)
        : "memory");
    asm volatile(
        "stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
        : "memory");

    out[tid] = smem[tid * 4];
}
#endif

int main() {
    constexpr int input_words = 192;
    constexpr int output_words = 128;

    uint32_t* h_in = static_cast<uint32_t*>(malloc(input_words * sizeof(uint32_t)));
    uint32_t* h_out = static_cast<uint32_t*>(malloc(output_words * sizeof(uint32_t)));

    for (int i = 0; i < input_words; ++i) {
        h_in[i] = 0x1000u + static_cast<uint32_t>(i);
    }

    uint32_t *d_in = nullptr;
    uint32_t *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, input_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, output_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, input_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, output_words * sizeof(uint32_t)));

    stmatrix_kernel<<<1, 32>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, output_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("first eight output words:");
    for (int i = 0; i < 8; ++i) {
        printf(" %08x", h_out[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
