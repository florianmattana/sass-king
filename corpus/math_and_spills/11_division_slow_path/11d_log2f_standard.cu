// sass-king step 11d: log2f standard (IEEE accurate)
// Expected SASS: CALL to math library helper for log2f
// Compile: nvcc -arch=sm_120 -o 11d_log2f_standard 11d_log2f_standard.cu
// Dump:    cuobjdump --dump-sass 11d_log2f_standard

#include <cmath>

__global__ void log2f_standard(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = log2f(a[i]);
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    log2f_standard<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}