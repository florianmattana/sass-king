// sass-king step 05f: both constants are special (0.5)
// Question: does ptxas inline both as immediates (no HFMA2)?
// Compile: nvcc -arch=sm_120 -o 05f_half_half 05f_half_half.cu

__global__ void vector_half_half(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * 0.5f + 0.5f;
    }
}

int main() {
    const int n = 1024;
    float *d_a, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    vector_half_half<<<(n + 255) / 256, 256>>>(d_a, d_c, n);
    cudaFree(d_a); cudaFree(d_c);
    return 0;
}