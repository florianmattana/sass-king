// sass-king step 04: vector kernel with runtime loop
// Delta from step 01: introduce a runtime-bounded for loop with non-foldable body
// Question: how does ptxas emit BRA backward, ISETP loop end, and the counter register?
// Compile: nvcc -arch=sm_120 -o 04_vector_loop 04_vector_loop.cu
// Dump:    cuobjdump --dump-sass 04_vector_loop

__global__ void vector_loop(const float* a, float* c, int n, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int k = 0; k < K; k++) {
            x = x * 1.001f + 0.5f;  // non-foldable: depends on previous x
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

    vector_loop<<<(n + 255) / 256, 256>>>(d_a, d_c, n, 100);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}