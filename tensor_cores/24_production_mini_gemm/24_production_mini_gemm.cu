// 24_production_mini_gemm.cu
//
// Chapter 24 production-like mini-GEMM audit probes.
//
// Compile example:
//   nvcc -arch=compute_120a -code=sm_120a -DVARIANT=2400 -o build/24a_minimal_hmma_tile 24_production_mini_gemm.cu

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
#define VARIANT 2400
#endif

__device__ __forceinline__ uint32_t smem_u32_ptr(void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16B(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr), "n"(16)
        : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
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

__device__ __forceinline__ void stmatrix_x4(uint32_t addr, uint32_t r0, uint32_t r1,
                                            uint32_t r2, uint32_t r3) {
    asm volatile(
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
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

__device__ __forceinline__ void qmma_e4m3(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                          uint32_t b0, uint32_t b1,
                                          float& d0, float& d1, float& d2, float& d3) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
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

__device__ __forceinline__ void sparse_omma_sf_4x(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
                                                  uint32_t meta, uint32_t sfa, uint32_t sfb,
                                                  float& d0, float& d1, float& d2, float& d3) {
    uint16_t bid = 0;
    uint16_t tid_sf = 0;
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, "
        "%16, 0x0, {%17}, {%18, %19}, {%20}, {%21, %22};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3), "r"(meta),
          "r"(sfa), "h"(bid), "h"(tid_sf), "r"(sfb), "h"(bid), "h"(tid_sf));
}

__device__ __noinline__ float cold_error_path(float x) {
    asm volatile("brkpt;\n" ::);
    return x + 1.0f;
}

__global__ void mini_gemm_kernel(const uint4* __restrict__ gmem,
                                 const uint32_t* __restrict__ params,
                                 float4* __restrict__ out,
                                 int n) {
    __shared__ __align__(16) uint32_t smem[2048];
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    uint32_t a0 = 0x22222222u + lane;
    uint32_t a1 = 0x33333333u + lane;
    uint32_t a2 = 0x44444444u + lane;
    uint32_t a3 = 0x55555555u + lane;
    uint32_t b0 = 0x66666666u + lane;
    uint32_t b1 = 0x77777777u + lane;
    uint32_t b2 = 0x11111111u + lane;
    uint32_t b3 = 0x99999999u + lane;
    uint32_t meta = 0xaaaaaaaau;
    uint32_t sfa = 0x38383838u;
    uint32_t sfb = 0x38383838u;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

#if VARIANT == 2404 || VARIANT == 2405 || VARIANT == 2419 || VARIANT == 2423
    cp_async_16B(smem_u32_ptr(&smem[tid * 4]), &gmem[tid]);
    cp_async_commit();
#if VARIANT == 2405 || VARIANT == 2419 || VARIANT == 2423
    cp_async_16B(smem_u32_ptr(&smem[512 + tid * 4]), &gmem[32 + tid]);
    cp_async_commit();
#endif
#if VARIANT == 2423
    cp_async_16B(smem_u32_ptr(&smem[1024 + tid * 4]), &gmem[64 + tid]);
    cp_async_commit();
    cp_async_wait<2>();
    cp_async_wait<1>();
#endif
    cp_async_wait<0>();
    __syncthreads();
    ldmatrix_x4(smem_u32_ptr(&smem[tid * 4]), a0, a1, a2, a3);
    ldmatrix_x2(smem_u32_ptr(&smem[512 + tid * 4]), b0, b1);
#endif

#if VARIANT == 2406
    ldmatrix_x2(smem_u32_ptr(&smem[512 + tid * 4]), b0, b1);
    ldmatrix_x4(smem_u32_ptr(&smem[tid * 4]), a0, a1, a2, a3);
#elif VARIANT == 2416
    ldmatrix_x4(smem_u32_ptr(&smem[tid * 4 + 8]), a0, a1, a2, a3);
    ldmatrix_x2(smem_u32_ptr(&smem[512 + tid * 4 + 8]), b0, b1);
#endif

#if VARIANT == 2421
    uint32_t uniformish = params[blockIdx.x & 3];
    asm volatile("mov.u32 %0, %1;\n" : "=r"(a0) : "r"(uniformish));
#elif VARIANT == 2422
    uint64_t offset = static_cast<uint64_t>(params[0]) * static_cast<uint64_t>(n + lane);
    a0 ^= static_cast<uint32_t>(offset);
    a1 ^= static_cast<uint32_t>(offset >> 32);
#elif VARIANT == 2426
    sfa = params[(lane & 3) + 0];
    sfb = params[(lane & 3) + 4];
#elif VARIANT == 2427
    meta = params[lane & 7];
#elif VARIANT == 2428
    int stride_m = static_cast<int>(params[0] & 31u) + 1;
    int stride_k = static_cast<int>(params[1] & 31u) + 1;
    a0 ^= static_cast<uint32_t>(lane * stride_m);
    b0 ^= static_cast<uint32_t>(lane * stride_k);
#endif

#if VARIANT == 2400 || VARIANT == 2404 || VARIANT == 2405 || VARIANT == 2406 || \
    VARIANT == 2411 || VARIANT == 2412 || VARIANT == 2413 || VARIANT == 2414 || \
    VARIANT == 2415 || VARIANT == 2416 || VARIANT == 2419 || VARIANT == 2420 || \
    VARIANT == 2421 || VARIANT == 2422 || VARIANT == 2423 || VARIANT == 2424 || \
    VARIANT == 2425 || VARIANT == 2428 || VARIANT == 2429
    hmma_step(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
#elif VARIANT == 2401
    qmma_e4m3(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
#elif VARIANT == 2402
    qmma_e2m1(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
#elif VARIANT == 2403 || VARIANT == 2426
    omma_sf_4x(a0, a1, a2, a3, b0, b1, sfa, sfb, d0, d1, d2, d3);
#elif VARIANT == 2407
    qmma_e4m3(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    qmma_e4m3(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    qmma_e4m3(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    qmma_e4m3(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
#elif VARIANT == 2409 || VARIANT == 2410
    qmma_e2m1(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    stmatrix_x4(smem_u32_ptr(&smem[tid * 4]), __float_as_uint(d0), __float_as_uint(d1),
                __float_as_uint(d2), __float_as_uint(d3));
    __syncthreads();
#elif VARIANT == 2417 || VARIANT == 2427
    sparse_qmma(a0, a1, a2, a3, b0, b1, b2, b3, meta, d0, d1, d2, d3);
#elif VARIANT == 2418
    sparse_omma_sf_4x(a0, a1, a2, a3, b0, b1, b2, b3, meta, sfa, sfb, d0, d1, d2, d3);
#endif

#if VARIANT == 2412
    #pragma unroll 1
    for (int k = 0; k < (n & 3) + 1; ++k) {
        hmma_step(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
#elif VARIANT == 2413
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        hmma_step(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
#elif VARIANT == 2414
    if (lane & 1) {
        hmma_step(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
    }
#elif VARIANT == 2415
    float regs[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) regs[i] = d0 + static_cast<float>(i + lane);
    d0 += regs[(lane + 3) & 15];
#elif VARIANT == 2425
    atomicAdd(reinterpret_cast<float*>(&out[0]), d0);
#elif VARIANT == 2429
    if (n < 0 && lane == 0) {
        d0 = cold_error_path(d0);
    }
#endif

#if VARIANT == 2410
    uint32_t x0 = smem[tid * 4 + 0];
    uint32_t x1 = smem[tid * 4 + 1];
    uint32_t x2 = smem[tid * 4 + 2];
    uint32_t x3 = smem[tid * 4 + 3];
    out[tid] = make_float4(__uint_as_float(x0), __uint_as_float(x1), __uint_as_float(x2), __uint_as_float(x3));
#elif VARIANT == 2411 || VARIANT == 2424
    if (tid < n) {
        out[tid] = make_float4(d0, d1, d2, d3);
    }
#elif VARIANT == 2408 || VARIANT == 2419 || VARIANT == 2420 || VARIANT == 2422 || VARIANT == 2428 || VARIANT == 2429
    out[tid] = make_float4(d0, d1, d2, d3);
#else
    out[tid] = make_float4(d0, d1, d2, d3);
#endif
}

int main() {
    constexpr int gmem_words = 128;
    constexpr int param_words = 32;
    constexpr int out_words = 32;

    uint4* h_gmem = static_cast<uint4*>(malloc(gmem_words * sizeof(uint4)));
    uint32_t* h_params = static_cast<uint32_t*>(malloc(param_words * sizeof(uint32_t)));
    float4* h_out = static_cast<float4*>(malloc(out_words * sizeof(float4)));

    for (int i = 0; i < gmem_words; ++i) {
        h_gmem[i] = make_uint4(0x22222222u + i, 0x33333333u + i, 0x44444444u + i, 0x55555555u + i);
    }
    for (int i = 0; i < param_words; ++i) h_params[i] = 0x38383838u + static_cast<uint32_t>(i);

    uint4* d_gmem = nullptr;
    uint32_t* d_params = nullptr;
    float4* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gmem, gmem_words * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&d_params, param_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, out_words * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(d_gmem, h_gmem, gmem_words * sizeof(uint4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_params, h_params, param_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, out_words * sizeof(float4)));

    mini_gemm_kernel<<<1, 32>>>(d_gmem, d_params, d_out, 16);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_words * sizeof(float4), cudaMemcpyDeviceToHost));

    const uint32_t* words = reinterpret_cast<const uint32_t*>(&h_out[0]);
    printf("variant=%d first output words:", VARIANT);
    for (int i = 0; i < 8; ++i) printf(" %08x", words[i]);
    printf("\n");

    cudaFree(d_gmem);
    cudaFree(d_params);
    cudaFree(d_out);
    free(h_gmem);
    free(h_params);
    free(h_out);
    return 0;
}
