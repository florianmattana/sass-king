// sass-king step 12e: loop with many live accumulators, natural register pressure
// 16 parallel accumulators across a loop
// Expected SASS: probably no spill if ptxas allocates enough registers (up to 255 available)
// Compile: nvcc -arch=sm_120 -o 12e_loop_acc 12e_loop_acc.cu
// Dump:    cuobjdump --dump-sass 12e_loop_acc

__global__ void loop_16acc(const float* a, float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    float s8 = 0.0f, s9 = 0.0f, sA = 0.0f, sB = 0.0f;
    float sC = 0.0f, sD = 0.0f, sE = 0.0f, sF = 0.0f;

    float v = a[i];

    for (int k = 0; k < iters; ++k) {
        float x = v + (float)k;
        s0 = fmaf(x, s1, s0);
        s1 = fmaf(x, s2, s1);
        s2 = fmaf(x, s3, s2);
        s3 = fmaf(x, s4, s3);
        s4 = fmaf(x, s5, s4);
        s5 = fmaf(x, s6, s5);
        s6 = fmaf(x, s7, s6);
        s7 = fmaf(x, s8, s7);
        s8 = fmaf(x, s9, s8);
        s9 = fmaf(x, sA, s9);
        sA = fmaf(x, sB, sA);
        sB = fmaf(x, sC, sB);
        sC = fmaf(x, sD, sC);
        sD = fmaf(x, sE, sD);
        sE = fmaf(x, sF, sE);
        sF = fmaf(x, s0, sF);
    }

    out[i] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
           + s8 + s9 + sA + sB + sC + sD + sE + sF;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_out, bytes);

    loop_16acc<<<(n + 255) / 256, 256>>>(d_a, d_out, n, 100);

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}