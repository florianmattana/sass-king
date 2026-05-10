// sass-king step 02: vector add + 1.0f
// Delta from step 01: one extra addition
// Compile: nvcc -arch=sm_120 -o 02_vector_add_plus1 02_vector_add_plus1.cu
// Dump:    cuobjdump --dump-sass 02_vector_add_plus1

__global__ void vector_add_plus1(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i] + 1.0f;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vector_add_plus1<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}