// 23_fragment_layout.cu
//
// Chapter 23 FP4 / FP6 fragment-layout probes.
//
// Compile examples:
//   nvcc -arch=compute_120a -code=sm_120a -DVARIANT=2300 -o build/23a_e2m1_pack_baseline 23_fragment_layout.cu
//   cuobjdump --dump-sass build/23a_e2m1_pack_baseline > 23a_e2m1_pack_baseline.sass

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
#define VARIANT 2300
#endif

__device__ __forceinline__ uint32_t smem_u32_ptr(void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t lane_pattern(const uint32_t tid, const uint32_t salt) {
    return (tid * 0x01010101u) ^ salt;
}

__device__ __forceinline__ void qmma_e2m1_e2m1(const uint32_t a0, const uint32_t a1,
                                               const uint32_t a2, const uint32_t a3,
                                               const uint32_t b0, const uint32_t b1,
                                               const float c0, const float c1,
                                               const float c2, const float c3,
                                               float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void qmma_e3m2_e3m2(const uint32_t a0, const uint32_t a1,
                                               const uint32_t a2, const uint32_t a3,
                                               const uint32_t b0, const uint32_t b1,
                                               const float c0, const float c1,
                                               const float c2, const float c3,
                                               float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void qmma_e2m3_e2m3(const uint32_t a0, const uint32_t a1,
                                               const uint32_t a2, const uint32_t a3,
                                               const uint32_t b0, const uint32_t b1,
                                               const float c0, const float c1,
                                               const float c2, const float c3,
                                               float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e2m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void qmma_e3m2_e2m3(const uint32_t a0, const uint32_t a1,
                                               const uint32_t a2, const uint32_t a3,
                                               const uint32_t b0, const uint32_t b1,
                                               const float c0, const float c1,
                                               const float c2, const float c3,
                                               float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e2m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void qmma_e2m3_e3m2(const uint32_t a0, const uint32_t a1,
                                               const uint32_t a2, const uint32_t a3,
                                               const uint32_t b0, const uint32_t b1,
                                               const float c0, const float c1,
                                               const float c2, const float c3,
                                               float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e3m2.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void qmma_sf_e4m3(const uint32_t a0, const uint32_t a1,
                                             const uint32_t a2, const uint32_t a3,
                                             const uint32_t b0, const uint32_t b1,
                                             const uint32_t sfa, const uint32_t sfb,
                                             const float c0, const float c1,
                                             const float c2, const float c3,
                                             float& d0, float& d1, float& d2, float& d3) {
    uint16_t bid = 0;
    uint16_t tid_sf = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"(bid), "h"(tid_sf), "r"(sfb), "h"(bid), "h"(tid_sf));
}

__device__ __forceinline__ void omma_sf_2x(const uint32_t a0, const uint32_t a1,
                                           const uint32_t a2, const uint32_t a3,
                                           const uint32_t b0, const uint32_t b1,
                                           const uint32_t sfa, const uint32_t sfb,
                                           const float c0, const float c1,
                                           const float c2, const float c3,
                                           float& d0, float& d1, float& d2, float& d3) {
    uint16_t bid = 0;
    uint16_t tid_sf = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"(bid), "h"(tid_sf), "r"(sfb), "h"(bid), "h"(tid_sf));
}

__device__ __forceinline__ void omma_sf_4x(const uint32_t a0, const uint32_t a1,
                                           const uint32_t a2, const uint32_t a3,
                                           const uint32_t b0, const uint32_t b1,
                                           const uint32_t sfa, const uint32_t sfb,
                                           const float c0, const float c1,
                                           const float c2, const float c3,
                                           float& d0, float& d1, float& d2, float& d3) {
    uint16_t bid = 0;
    uint16_t tid_sf = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "h"(bid), "h"(tid_sf), "r"(sfb), "h"(bid), "h"(tid_sf));
}

__device__ __forceinline__ void sparse_qmma_sf(const uint32_t a0, const uint32_t a1,
                                               const uint32_t a2, const uint32_t a3,
                                               const uint32_t b0, const uint32_t b1,
                                               const uint32_t b2, const uint32_t b3,
                                               const uint32_t sfa, const uint32_t sfb,
                                               const uint32_t meta,
                                               const float c0, const float c1,
                                               const float c2, const float c3,
                                               float& d0, float& d1, float& d2, float& d3) {
    uint16_t bid = 0;
    uint16_t tid_sf = 0;
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, "
        "%16, 0x0, {%17}, {%18, %19}, {%20}, {%21, %22};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(meta), "r"(sfa), "h"(bid), "h"(tid_sf), "r"(sfb), "h"(bid), "h"(tid_sf));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

__global__ void fragment_layout_kernel(const uint32_t* __restrict__ in,
                                       uint32_t* __restrict__ out) {
    __shared__ __align__(16) uint32_t smem[1024];
    const uint32_t tid = threadIdx.x;

    uint32_t a0 = in[tid * 8 + 0];
    uint32_t a1 = in[tid * 8 + 1];
    uint32_t a2 = in[tid * 8 + 2];
    uint32_t a3 = in[tid * 8 + 3];
    uint32_t b0 = in[tid * 8 + 4];
    uint32_t b1 = in[tid * 8 + 5];
    uint32_t sfa = in[tid * 8 + 6];
    uint32_t sfb = in[tid * 8 + 7];
    uint32_t meta = 0x0u;

    float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

#if VARIANT == 2305
    a0 = lane_pattern(tid, 0x11111111u); a1 = lane_pattern(tid, 0x22222222u);
    a2 = lane_pattern(tid, 0x44444444u); a3 = lane_pattern(tid, 0x88888888u);
    b0 = lane_pattern(tid, 0x12121212u); b1 = lane_pattern(tid, 0x24242424u);
#elif VARIANT == 2306
    a0 = lane_pattern(tid, 0x03030303u); a1 = lane_pattern(tid, 0x13131313u);
    a2 = lane_pattern(tid, 0x23232323u); a3 = lane_pattern(tid, 0x33333333u);
    b0 = lane_pattern(tid, 0x43434343u); b1 = lane_pattern(tid, 0x53535353u);
#elif VARIANT == 2307
    a0 = lane_pattern(tid, 0x06060606u); a1 = lane_pattern(tid, 0x16161616u);
    a2 = lane_pattern(tid, 0x26262626u); a3 = lane_pattern(tid, 0x36363636u);
    b0 = lane_pattern(tid, 0x46464646u); b1 = lane_pattern(tid, 0x56565656u);
#elif VARIANT == 2314
    a0 ^= 0x80000000u; a1 ^= 0x00800000u; a2 ^= 0x00008000u; a3 ^= 0x00000080u;
#elif VARIANT == 2318
    a0 = (tid == 15) ? 0xaaaaaaaau : 0x11111111u;
    a1 = (tid == 16) ? 0xbbbbbbbbu : 0x22222222u;
    a2 = (tid == 31) ? 0xccccccccu : 0x33333333u;
    a3 = (tid == 0)  ? 0xddddddddu : 0x44444444u;
#elif VARIANT == 2319
    a0 = 0x0000000fu; a1 = 0x000000f0u; a2 = 0x00000f00u; a3 = 0x0000f000u;
    b0 = 0x000f0000u; b1 = 0x00f00000u;
#elif VARIANT == 2320
    a0 = 0x00000000u; a1 = 0x7f7f7f7fu; a2 = 0xffffffffu; a3 = 0x80808080u;
    b0 = 0x38383838u; b1 = 0x77777777u;
#endif

#if VARIANT == 2309 || VARIANT == 2321
    const uint32_t base = (VARIANT == 2321) ? 8u : 0u;
    smem[tid * 4 + base + 0] = a0;
    smem[tid * 4 + base + 1] = a1;
    smem[tid * 4 + base + 2] = a2;
    smem[tid * 4 + base + 3] = a3;
    smem[512 + tid * 4 + base + 0] = b0;
    smem[512 + tid * 4 + base + 1] = b1;
    __syncthreads();
    ldmatrix_x4(smem_u32_ptr(&smem[tid * 4 + base]), a0, a1, a2, a3);
    ldmatrix_x2(smem_u32_ptr(&smem[512 + tid * 4 + base]), b0, b1);
#elif VARIANT == 2310
    a0 = 0x01234567u ^ tid; a1 = 0x89abcdefu ^ (tid << 1);
    a2 = 0xfedcba98u ^ (tid << 2); a3 = 0x76543210u ^ (tid << 3);
    b0 = 0x13579bdfu ^ (tid << 4); b1 = 0x2468ace0u ^ (tid << 5);
#endif

#if VARIANT == 2300 || VARIANT == 2305 || VARIANT == 2309 || VARIANT == 2310 || VARIANT == 2314 || \
    VARIANT == 2318 || VARIANT == 2319 || VARIANT == 2320 || VARIANT == 2321
    qmma_e2m1_e2m1(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3);
#elif VARIANT == 2301 || VARIANT == 2306
    qmma_e3m2_e3m2(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3);
#elif VARIANT == 2302 || VARIANT == 2307
    qmma_e2m3_e2m3(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3);
#elif VARIANT == 2303 || VARIANT == 2317
    qmma_e3m2_e2m3(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3);
#elif VARIANT == 2304
    qmma_e2m3_e3m2(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3);
#elif VARIANT == 2308
    qmma_sf_e4m3(a0, a1, a2, a3, b0, b1, sfa, sfb, c0, c1, c2, c3, d0, d1, d2, d3);
#elif VARIANT == 2315
    omma_sf_4x(a0, a1, a2, a3, b0, b1, sfb, sfb, c0, c1, c2, c3, d0, d1, d2, d3);
#elif VARIANT == 2311
    qmma_e2m1_e2m1(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3);
#elif VARIANT == 2313
    qmma_e3m2_e2m3(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, d0, d1, d2, d3);
#elif VARIANT == 2316
    sparse_qmma_sf(a0, a1, a2, a3, b0, b1, a2, a3, sfa, sfb, meta, c0, c1, c2, c3, d0, d1, d2, d3);
#else
#error "Unknown VARIANT"
#endif

    out[tid * 12 + 0] = __float_as_uint(d0);
    out[tid * 12 + 1] = __float_as_uint(d1);
    out[tid * 12 + 2] = __float_as_uint(d2);
    out[tid * 12 + 3] = __float_as_uint(d3);
    out[tid * 12 + 4] = a0;
    out[tid * 12 + 5] = a1;
    out[tid * 12 + 6] = a2;
    out[tid * 12 + 7] = a3;
    out[tid * 12 + 8] = b0;
    out[tid * 12 + 9] = b1;
    out[tid * 12 + 10] = sfa;
    out[tid * 12 + 11] = sfb;
}

#ifdef TEST_INVALID_FORMAT
__global__ void invalid_format_kernel(uint32_t* out) {
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    out[0] = __float_as_uint(d0) ^ __float_as_uint(d1) ^ __float_as_uint(d2) ^ __float_as_uint(d3);
}
#endif

int main() {
    constexpr int input_words = 32 * 8;
    constexpr int output_words = 32 * 12;
    uint32_t* h_in = static_cast<uint32_t*>(malloc(input_words * sizeof(uint32_t)));
    uint32_t* h_out = static_cast<uint32_t*>(malloc(output_words * sizeof(uint32_t)));

    for (int lane = 0; lane < 32; ++lane) {
        h_in[lane * 8 + 0] = 0x22222222u;
        h_in[lane * 8 + 1] = 0x33333333u;
        h_in[lane * 8 + 2] = 0x44444444u;
        h_in[lane * 8 + 3] = 0x55555555u;
        h_in[lane * 8 + 4] = 0x66666666u;
        h_in[lane * 8 + 5] = 0x77777777u;
        h_in[lane * 8 + 6] = 0x00007f7fu;
        h_in[lane * 8 + 7] = 0x38383838u;
    }

    uint32_t* d_in = nullptr;
    uint32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, input_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, output_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, input_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, output_words * sizeof(uint32_t)));

    fragment_layout_kernel<<<1, 32>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, output_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("variant=%d first output words:", VARIANT);
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
