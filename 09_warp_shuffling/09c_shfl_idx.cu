// sass-king step 09c: warp broadcast via __shfl_sync
// Delta from 09b: SHFL.IDX instead of SHFL.BFLY, broadcast value of lane 0 to all
// Goal: observe SHFL.IDX opcode and compare to SHFL.BFLY
// Compile: nvcc -arch=sm_120 -o 09c_shfl_idx 09c_shfl_idx.cu
// Dump:    cuobjdump --dump-sass 09c_shfl_idx

__global__ void shfl_idx_broadcast(const float* a, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[i];

    // Every thread receives the value from lane 0 of its warp
    float broadcast = __shfl_sync(0xffffffff, val, 0);

    c[i] = val + broadcast;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    shfl_idx_broadcast<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}