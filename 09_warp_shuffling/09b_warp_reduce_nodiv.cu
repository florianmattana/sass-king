// sass-king step 09b: warp reduction without bounds check
// Delta from 09a: removed the if (i < n) divergence before SHFL
// Goal: test if BSSY/BSYNC.RECONVERGENT disappear when no divergence precedes SHFL
// Compile: nvcc -arch=sm_120 -o 09b_warp_reduce_nodiv 09b_warp_reduce_nodiv.cu
// Dump:    cuobjdump --dump-sass 09b_warp_reduce_nodiv

__global__ void warp_reduce_nodiv(const float* a, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[i];

    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);

    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = val;
    }
}

int main() {
    const int n = 1024;
    const int bytes_in = n * sizeof(float);
    const int bytes_out = (n / 32) * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes_in);
    cudaMalloc(&d_c, bytes_out);

    warp_reduce_nodiv<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}