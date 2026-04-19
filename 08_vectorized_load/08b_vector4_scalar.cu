// sass-king step 08b: forced vectorization pattern with scalar pointer
// Delta from kernel 08a: same data layout but accessed via float* with manual indexing
// Goal: test if ptxas recognizes the 4-contiguous-float pattern and auto-vectorizes
//       to LDG.E.128, or emits 4 separate LDG.E instructions
// Compile: nvcc -arch=sm_120 -o 08b_vector4_scalar 08b_vector4_scalar.cu
// Dump:    cuobjdump --dump-sass 08b_vector4_scalar

__global__ void vector4_scalar(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int base = i * 4;
    if (base + 3 < n) {
        float a0 = a[base + 0];
        float a1 = a[base + 1];
        float a2 = a[base + 2];
        float a3 = a[base + 3];

        float b0 = b[base + 0];
        float b1 = b[base + 1];
        float b2 = b[base + 2];
        float b3 = b[base + 3];

        c[base + 0] = a0 + b0;
        c[base + 1] = a1 + b1;
        c[base + 2] = a2 + b2;
        c[base + 3] = a3 + b3;
    }
}

int main() {
    const int n = 4096;  // 4096 floats = 1024 groups of 4
    const int bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vector4_scalar<<<(n / 4 + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}