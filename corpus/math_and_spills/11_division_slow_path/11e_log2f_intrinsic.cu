// sass-king step 11e: __log2f intrinsic (fast, less precise)
// Expected SASS: MUFU.LG2 direct, no CALL
// Compile: nvcc -arch=sm_120 -o 11e_log2f_intrinsic 11e_log2f_intrinsic.cu
// Dump:    cuobjdump --dump-sass 11e_log2f_intrinsic

__global__ void log2f_intrinsic(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = __log2f(a[i]);
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    log2f_intrinsic<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}