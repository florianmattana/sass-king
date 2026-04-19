// sass-king step 12g: kernel with 12 array pointers to observe pointer-spill behavior
// Each pointer costs LDC.64 + IMAD.WIDE and must stay live during LDG dispatch
// Expected SASS: possibly spill of pointer registers if ptxas can't keep all 12 live
// Compile: nvcc -arch=sm_120 -o 12g_many_ptrs 12g_many_ptrs.cu
// Dump:    cuobjdump --dump-sass 12g_many_ptrs

__global__ void many_ptrs(
    const float* a0, const float* a1, const float* a2, const float* a3,
    const float* a4, const float* a5, const float* a6, const float* a7,
    const float* a8, const float* a9, const float* aA, const float* aB,
    float* out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float s = 0.0f;
    s += a0[i];
    s += a1[i];
    s += a2[i];
    s += a3[i];
    s += a4[i];
    s += a5[i];
    s += a6[i];
    s += a7[i];
    s += a8[i];
    s += a9[i];
    s += aA[i];
    s += aB[i];

    out[i] = s;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_arrays[12], *d_out;
    for (int k = 0; k < 12; ++k) cudaMalloc(&d_arrays[k], bytes);
    cudaMalloc(&d_out, bytes);

    many_ptrs<<<(n + 255) / 256, 256>>>(
        d_arrays[0], d_arrays[1], d_arrays[2], d_arrays[3],
        d_arrays[4], d_arrays[5], d_arrays[6], d_arrays[7],
        d_arrays[8], d_arrays[9], d_arrays[10], d_arrays[11],
        d_out, n
    );

    for (int k = 0; k < 12; ++k) cudaFree(d_arrays[k]);
    cudaFree(d_out);
    return 0;
}