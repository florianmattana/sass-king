// sass-king step 11c: integer division by runtime variable (s32 signed)
// Expected SASS: CALL to signed version, or unsigned helper + sign handling inline
// Compile: nvcc -arch=sm_120 -o 11c_div_s32_runtime 11c_div_s32_runtime.cu
// Dump:    cuobjdump --dump-sass 11c_div_s32_runtime

__global__ void div_s32_runtime(const int* a, int* c, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] / d;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(int);

    int *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    div_s32_runtime<<<(n + 255) / 256, 256>>>(d_a, d_c, n, 7);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}