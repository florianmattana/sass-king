// sass-king step 09e: explicit __syncwarp()
// Delta from 09b: insert __syncwarp() between iterations
// Goal: observe what SASS __syncwarp() produces (if anything)
// Compile: nvcc -arch=sm_120 -o 09e_syncwarp 09e_syncwarp.cu
// Dump:    cuobjdump --dump-sass 09e_syncwarp

__global__ void sync_warp(const float* a, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[i];

    __syncwarp();
    val += __shfl_xor_sync(0xffffffff, val, 16);
    __syncwarp();
    val += __shfl_xor_sync(0xffffffff, val, 8);
    __syncwarp();
    val += __shfl_xor_sync(0xffffffff, val, 4);
    __syncwarp();
    val += __shfl_xor_sync(0xffffffff, val, 2);
    __syncwarp();
    val += __shfl_xor_sync(0xffffffff, val, 1);
    __syncwarp();

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

    sync_warp<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}