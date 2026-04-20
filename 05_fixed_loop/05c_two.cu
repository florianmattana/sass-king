// sass-king step 06b: vector loop fixed, constant = 2.0f
// Very simple bit pattern (0x40000000), all low bits zero
// Question: does ptxas use a special encoding for round numbers?
// Compile: nvcc -arch=sm_120 -o 06b_const_two 06b_const_two.cu

__global__ void vector_loop_two(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        for (int k = 0; k < 8; k++) {
            x = x * 2.0f + 0.5f;
        }
        c[i] = x;
    }
}

int main() {
    const int n = 1024;
    float *d_a, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    vector_loop_two<<<(n + 255) / 256, 256>>>(d_a, d_c, n);
    cudaFree(d_a); cudaFree(d_c);
    return 0;
}