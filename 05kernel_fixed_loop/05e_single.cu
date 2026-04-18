// sass-king step 05e: single use of constant
// Delta from step 05: no loop, constant used exactly once
// Question: does ptxas still pre-materialize the constant via HFMA2,
// or does it inline it as immediate in the FFMA?
// Compile: nvcc -arch=sm_120 -o 05e_single 05e_single.cu

__global__ void vector_single(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * 1.001f + 0.5f;   // single FFMA, constant used once
    }
}

int main() {
    const int n = 1024;
    float *d_a, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    vector_single<<<(n + 255) / 256, 256>>>(d_a, d_c, n);
    cudaFree(d_a); cudaFree(d_c);
    return 0;
}