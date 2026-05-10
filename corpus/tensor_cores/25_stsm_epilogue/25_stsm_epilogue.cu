// 25_stsm_epilogue.cu
//
// Chapter 25 STSM epilogue layout and storeback probes.
//
// Compile example:
//   nvcc -arch=compute_120a -code=sm_120a -DVARIANT=2500 -o build/25a_stsm_x1_runtime_layout 25_stsm_epilogue.cu

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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
#define VARIANT 2500
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

__device__ __forceinline__ void stmatrix_x4(uint32_t addr, uint32_t r0, uint32_t r1,
                                            uint32_t r2, uint32_t r3) {
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

__device__ __forceinline__ void stmatrix_x4_trans(uint32_t addr, uint32_t r0, uint32_t r1,
                                                  uint32_t r2, uint32_t r3) {
    asm volatile(
        "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
        : "memory");
}

__device__ __forceinline__ void hmma_step(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1,
                                          float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
}

__device__ __forceinline__ void qmma_e2m1(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1,
                                          float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
}

__device__ __forceinline__ void omma_sf_4x(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1,
                                           uint32_t sfa, uint32_t sfb,
                                           float& d0, float& d1, float& d2, float& d3) {
    uint16_t bid = 0;
    uint16_t tid_sf = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(sfa), "h"(bid), "h"(tid_sf), "r"(sfb), "h"(bid), "h"(tid_sf));
}

__device__ __forceinline__ void sparse_qmma(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
                                            uint32_t meta,
                                            float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3), "r"(meta));
}

__device__ __forceinline__ uint32_t pack_f32_to_f16(float lo, float hi) {
    uint32_t lo_bits = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(lo)));
    uint32_t hi_bits = static_cast<uint32_t>(__half_as_ushort(__float2half_rn(hi)));
    return lo_bits | (hi_bits << 16);
}

__device__ __forceinline__ uint32_t pack_f32_to_bf16(float lo, float hi) {
    uint32_t lo_bits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(lo)));
    uint32_t hi_bits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(hi)));
    return lo_bits | (hi_bits << 16);
}

__device__ __forceinline__ uint32_t lane_pattern(int lane, int slot) {
    return 0x25000000u | (static_cast<uint32_t>(slot & 0xff) << 16) | static_cast<uint32_t>(lane);
}

__global__ void stsm_epilogue_kernel(const uint32_t* __restrict__ in,
                                     uint32_t* __restrict__ out,
                                     int n) {
    __shared__ __align__(16) uint32_t smem[4096];
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    uint32_t r0 = in[tid + 0] ^ lane_pattern(lane, 0);
    uint32_t r1 = in[tid + 32] ^ lane_pattern(lane, 1);
    uint32_t r2 = in[tid + 64] ^ lane_pattern(lane, 2);
    uint32_t r3 = in[tid + 96] ^ lane_pattern(lane, 3);
    uint32_t b0 = in[tid + 128] ^ lane_pattern(lane, 4);
    uint32_t b1 = in[tid + 160] ^ lane_pattern(lane, 5);
    uint32_t b2 = in[tid + 192] ^ lane_pattern(lane, 6);
    uint32_t b3 = in[tid + 224] ^ lane_pattern(lane, 7);
    uint32_t meta = 0xaaaaaaaau ^ static_cast<uint32_t>(lane);
    uint32_t sfa = 0x38383838u;
    uint32_t sfb = 0x38383838u;

    int base = tid * 4;
#if VARIANT == 2511
    base = tid * 4 + 8;
#elif VARIANT == 2523
    base = tid * 8;
#endif
    uint32_t addr = smem_u32_ptr(&smem[base]);

    float d0 = static_cast<float>(lane) + 1.0f;
    float d1 = static_cast<float>(lane) + 2.0f;
    float d2 = static_cast<float>(lane) + 3.0f;
    float d3 = static_cast<float>(lane) + 4.0f;

#if VARIANT == 2506 || VARIANT == 2508 || VARIANT == 2510 || VARIANT == 2512 || \
    VARIANT == 2513 || VARIANT == 2514 || VARIANT == 2517 || VARIANT == 2518 || \
    VARIANT == 2519 || VARIANT == 2522 || VARIANT == 2524 || VARIANT == 2525
    hmma_step(r0, r1, r2, r3, b0, b1, d0, d1, d2, d3);
#elif VARIANT == 2507
    qmma_e2m1(r0, r1, r2, r3, b0, b1, d0, d1, d2, d3);
#elif VARIANT == 2520
    omma_sf_4x(r0, r1, r2, r3, b0, b1, sfa, sfb, d0, d1, d2, d3);
#elif VARIANT == 2521
    sparse_qmma(r0, r1, r2, r3, b0, b1, b2, b3, meta, d0, d1, d2, d3);
#endif

#if VARIANT == 2518
    r0 = pack_f32_to_f16(d0, d1);
    r1 = pack_f32_to_f16(d2, d3);
    r2 = pack_f32_to_f16(d0 + 1.0f, d1 + 1.0f);
    r3 = pack_f32_to_f16(d2 + 1.0f, d3 + 1.0f);
#elif VARIANT == 2519
    r0 = pack_f32_to_bf16(d0, d1);
    r1 = pack_f32_to_bf16(d2, d3);
    r2 = pack_f32_to_bf16(d0 + 1.0f, d1 + 1.0f);
    r3 = pack_f32_to_bf16(d2 + 1.0f, d3 + 1.0f);
#elif VARIANT == 2506 || VARIANT == 2507 || VARIANT == 2508 || VARIANT == 2510 || \
      VARIANT == 2512 || VARIANT == 2513 || VARIANT == 2514 || VARIANT == 2517 || \
      VARIANT == 2520 || VARIANT == 2521 || VARIANT == 2522 || VARIANT == 2524 || \
      VARIANT == 2525
    r0 = __float_as_uint(d0);
    r1 = __float_as_uint(d1);
    r2 = __float_as_uint(d2);
    r3 = __float_as_uint(d3);
#endif

#if VARIANT == 2524
    float regs[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        regs[i] = d0 + d1 + static_cast<float>(i + lane);
    }
    r0 ^= __float_as_uint(regs[(lane + 1) & 15]);
    r1 ^= __float_as_uint(regs[(lane + 5) & 15]);
    r2 ^= __float_as_uint(regs[(lane + 9) & 15]);
    r3 ^= __float_as_uint(regs[(lane + 13) & 15]);
#endif

#if VARIANT == 2500
    stmatrix_x1(addr, r0);
#elif VARIANT == 2501
    stmatrix_x2(addr, r0, r1);
#elif VARIANT == 2502 || VARIANT == 2506 || VARIANT == 2507 || VARIANT == 2508 || \
      VARIANT == 2510 || VARIANT == 2511 || VARIANT == 2512 || VARIANT == 2513 || \
      VARIANT == 2514 || VARIANT == 2515 || VARIANT == 2517 || VARIANT == 2518 || \
      VARIANT == 2519 || VARIANT == 2520 || VARIANT == 2521 || VARIANT == 2522 || \
      VARIANT == 2523 || VARIANT == 2524 || VARIANT == 2525
    stmatrix_x4(addr, r0, r1, r2, r3);
#elif VARIANT == 2503
    stmatrix_x1_trans(addr, r0);
#elif VARIANT == 2504
    stmatrix_x2_trans(addr, r0, r1);
#elif VARIANT == 2505
    stmatrix_x4_trans(addr, r0, r1, r2, r3);
#elif VARIANT == 2509
    smem[base + 0] = r0;
    smem[base + 1] = r1;
    smem[base + 2] = r2;
    smem[base + 3] = r3;
#else
#error "Unknown VARIANT"
#endif

#if VARIANT == 2514
    stmatrix_x4(smem_u32_ptr(&smem[base + 8]),
                 r0 ^ 0x01010101u, r1 ^ 0x02020202u,
                 r2 ^ 0x04040404u, r3 ^ 0x08080808u);
#endif

#if VARIANT == 2513
    out[tid] = smem[base];
#else
    __syncthreads();
#if VARIANT == 2510
    if (tid < n) {
        out[tid * 4 + 0] = smem[base + 0];
        out[tid * 4 + 1] = smem[base + 1];
        out[tid * 4 + 2] = smem[base + 2];
        out[tid * 4 + 3] = smem[base + 3];
    }
#elif VARIANT == 2514
    out[tid * 8 + 0] = smem[base + 0];
    out[tid * 8 + 1] = smem[base + 1];
    out[tid * 8 + 2] = smem[base + 2];
    out[tid * 8 + 3] = smem[base + 3];
    out[tid * 8 + 4] = smem[base + 8];
    out[tid * 8 + 5] = smem[base + 9];
    out[tid * 8 + 6] = smem[base + 10];
    out[tid * 8 + 7] = smem[base + 11];
#elif VARIANT == 2522
    out[tid * 8 + 0] = smem[base + 0];
    out[tid * 8 + 2] = smem[base + 1];
    out[tid * 8 + 4] = smem[base + 2];
    out[tid * 8 + 6] = smem[base + 3];
#elif VARIANT == 2515 || VARIANT == 2525
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[tid * 8 + i] = smem[base + i];
    }
#else
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[tid * 4 + i] = smem[base + i];
    }
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
    constexpr int input_words = 256;
    constexpr int output_words = 512;

#ifdef TEST_UNSUPPORTED_STMATRIX_B8
    uint32_t* d_negative_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_negative_out, 128 * sizeof(uint32_t)));
    unsupported_stmatrix_b8_kernel<<<1, 32>>>(d_negative_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_negative_out);
    return 0;
#endif

    uint32_t* h_in = static_cast<uint32_t*>(malloc(input_words * sizeof(uint32_t)));
    uint32_t* h_out = static_cast<uint32_t*>(malloc(output_words * sizeof(uint32_t)));

    for (int i = 0; i < input_words; ++i) {
        h_in[i] = 0x10000000u + static_cast<uint32_t>(i * 0x101u);
    }

    uint32_t* d_in = nullptr;
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, input_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, output_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, input_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, output_words * sizeof(uint32_t)));

    stsm_epilogue_kernel<<<1, 32>>>(d_in, d_out, 16);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, output_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("variant=%d first output words:", VARIANT);
    for (int i = 0; i < 16; ++i) {
        printf(" %08x", h_out[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
