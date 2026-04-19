// sass-king step 11j: __fdividef fast division intrinsic (FP32)
// Expected SASS: MUFU.RCP + FMUL, no CALL
// Compile: nvcc -arch=sm_120 -o 11j_fdividef 11j_fdividef.cu
// Dump:    cuobjdump --dump-sass 11j_fdividef

__global__ void fdividef_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = __fdividef(a[i], b[i]);
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    fdividef_kernel<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}