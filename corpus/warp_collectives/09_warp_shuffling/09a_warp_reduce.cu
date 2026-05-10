// sass-king step 09a: warp-level reduction via __shfl_xor_sync
// Delta from earlier kernels: introduces warp primitives SHFL
// Goal: observe SHFL.BFLY pattern for butterfly reduction
// Compile: nvcc -arch=sm_120 -o 09a_warp_reduce 09a_warp_reduce.cu
// Dump:    cuobjdump --dump-sass 09a_warp_reduce

__global__ void warp_reduce(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? a[i] : 0.0f;

    // Butterfly reduction within warp (32 threads, 5 stages)
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);

    // Lane 0 of each warp writes the partial sum
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

    warp_reduce<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}