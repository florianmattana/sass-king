// sass-king step 09f: SHFL.UP variant
// Delta from 09c: __shfl_up_sync instead of __shfl_sync
// Goal: observe SHFL.UP opcode (lane N reads from lane N-delta)
// Compile: nvcc -arch=sm_120 -o 09f_shfl_up 09f_shfl_up.cu
// Dump:    cuobjdump --dump-sass 09f_shfl_up

__global__ void shfl_up(const float* a, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[i];

    // Each thread receives the value from the lane 4 positions earlier
    float shifted = __shfl_up_sync(0xffffffff, val, 4);

    c[i] = val + shifted;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    shfl_up<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}