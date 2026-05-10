// sass-king step 09g: SHFL.DOWN variant
// Delta from 09f: __shfl_down_sync instead of up
// Goal: observe SHFL.DOWN opcode (lane N reads from lane N+delta)
// Compile: nvcc -arch=sm_120 -o 09g_shfl_down 09g_shfl_down.cu
// Dump:    cuobjdump --dump-sass 09g_shfl_down

__global__ void shfl_down(const float* a, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[i];

    float shifted = __shfl_down_sync(0xffffffff, val, 4);

    c[i] = val + shifted;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    shfl_down<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}