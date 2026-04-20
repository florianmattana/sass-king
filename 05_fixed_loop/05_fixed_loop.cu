// sass-king step 05: vector kernel with COMPILE-TIME bounded loop
// Delta from step 04: K is now a compile-time constant (not a runtime arg)
// Question: does ptxas fully unroll, or does it generate a different pattern?
// Compile: nvcc -arch=sm_120 -o 05_vector_loop_fixed 05_vector_loop_fixed.cu
// Dump:    cuobjdump --dump-sass 05_vector_loop_fixed

__global__ void vector_loop_fixed(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int k = 0; k < 8; k++) {       // K = 8, known at compile time
            x = x * 1.001f + 0.5f;
        }
        c[i] = x;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    vector_loop_fixed<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}