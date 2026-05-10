// sass-king step 12b: same source as 12a, compiled with -maxrregcount=32
// Expected SASS: possibly mild spill pressure, depending on 12a's natural register count
// Compile: nvcc -arch=sm_120 -maxrregcount=32 -o 12b_maxreg32 12b_maxreg32.cu
// Dump:    cuobjdump --dump-sass 12b_maxreg32

__global__ void fma_chain_r32(const float* a, const float* b, const float* c, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x0 = a[i], x1 = b[i], x2 = c[i];
    float x3 = a[i] + 1.0f, x4 = b[i] + 2.0f, x5 = c[i] + 3.0f;
    float x6 = x0 * x1, x7 = x2 * x3, x8 = x4 * x5, x9 = x0 * x5;

    float r0 = x0 * x1 + x2;
    float r1 = x3 * x4 + x5;
    float r2 = x6 * x7 + x8;
    float r3 = x9 * x0 + x1;
    float r4 = r0 * r1 + r2;
    float r5 = r3 * r0 + r1;
    float r6 = r2 * r3 + r4;
    float r7 = r4 * r5 + r6;
    float r8 = r5 * r6 + r7;
    float r9 = r6 * r7 + r8;

    out[i] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_out, bytes);

    fma_chain_r32<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, d_out, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);
    return 0;
}