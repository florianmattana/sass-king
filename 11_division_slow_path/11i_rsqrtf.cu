// sass-king step 11i: rsqrtf (reciprocal square root)
// Expected SASS: MUFU.RSQ direct
// Compile: nvcc -arch=sm_120 -o 11i_rsqrtf 11i_rsqrtf.cu
// Dump:    cuobjdump --dump-sass 11i_rsqrtf

#include <cmath>

__global__ void rsqrtf_kernel(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = rsqrtf(a[i]);
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    rsqrtf_kernel<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}