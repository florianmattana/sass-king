// sass-king step 11h: sqrtf standard (IEEE accurate)
// Expected SASS: possibly MUFU.SQRT direct, or CALL to helper for IEEE compliance
// Compile: nvcc -arch=sm_120 -o 11h_sqrtf_standard 11h_sqrtf_standard.cu
// Dump:    cuobjdump --dump-sass 11h_sqrtf_standard

#include <cmath>

__global__ void sqrtf_standard(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = sqrtf(a[i]);
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    sqrtf_standard<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}