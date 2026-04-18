// sass-king step 06a: vector loop fixed, constant = 3.14159f
// Delta from step 05: change the multiplied constant
// Question: does ptxas still use HFMA2 magic, or fall back to MOV/IMAD?
// Compile: nvcc -arch=sm_120 -o 06a_const_pi 06a_const_pi.cu

__global__ void vector_loop_pi(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int k = 0; k < 8; k++) {
            x = x * 3.14159f + 0.5f;
        }
        c[i] = x;
    }
}

int main() {
    const int n = 1024;
    float *d_a, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    vector_loop_pi<<<(n + 255) / 256, 256>>>(d_a, d_c, n);
    cudaFree(d_a); cudaFree(d_c);
    return 0;
}