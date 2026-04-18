// sass-king step 06c: vector loop fixed, constant = 1e20f
// Large value, bit pattern 0x60AD78EC
// Question: does ptxas behavior change with magnitude?
// Compile: nvcc -arch=sm_120 -o 06c_const_large 06c_const_large.cu

__global__ void vector_loop_large(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int k = 0; k < 8; k++) {
            x = x * 1e20f + 0.5f;
        }
        c[i] = x;
    }
}

int main() {
    const int n = 1024;
    float *d_a, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    vector_loop_large<<<(n + 255) / 256, 256>>>(d_a, d_c, n);
    cudaFree(d_a); cudaFree(d_c);
    return 0;
}