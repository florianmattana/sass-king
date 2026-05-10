// sass-king step 03: vector fma (a*b+c)
// Delta from step 01: multiply instead of add, extra operand
// Question: does ptxas fuse FMUL+FADD into FFMA?
// Compile: nvcc -arch=sm_120 -o 03_vector_fma 03_vector_fma.cu
// Dump:    cuobjdump --dump-sass 03_vector_fma

__global__ void vector_fma(const float* a, const float* b, const float* c, float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d[i] = a[i] * b[i] + c[i];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_d, bytes);

    vector_fma<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, d_d, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    return 0;
}