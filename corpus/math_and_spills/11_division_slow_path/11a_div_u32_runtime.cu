// sass-king step 11a: integer division by runtime variable (u32)
// Expected SASS: CALL to __cuda_sm20_div_u32 or similar helper
// Compile: nvcc -arch=sm_120 -o 11a_div_u32_runtime 11a_div_u32_runtime.cu
// Dump:    cuobjdump --dump-sass 11a_div_u32_runtime

__global__ void div_u32_runtime(const unsigned int* a, unsigned int* c, int n, unsigned int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] / d;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(unsigned int);

    unsigned int *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    div_u32_runtime<<<(n + 255) / 256, 256>>>(d_a, d_c, n, 7);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}