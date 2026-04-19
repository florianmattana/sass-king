// sass-king step 12j: __noinline__ function with 16 float arguments
// Forces spill to local memory for argument passing
// Compile: nvcc -arch=sm_120 -o 12j_many_args 12j_many_args.cu
// Dump:    cuobjdump --dump-sass 12j_many_args

__noinline__ __device__ float combine16(
    float a0, float a1, float a2, float a3,
    float a4, float a5, float a6, float a7,
    float a8, float a9, float aA, float aB,
    float aC, float aD, float aE, float aF
) {
    float p0 = fmaf(a0, a1, a2);
    float p1 = fmaf(a3, a4, a5);
    float p2 = fmaf(a6, a7, a8);
    float p3 = fmaf(a9, aA, aB);
    float p4 = fmaf(aC, aD, aE);
    float p5 = p4 * aF;
    return p0 + p1 + p2 + p3 + p5;
}

__global__ void many_args(const float* a, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = a[i];

    float r = combine16(
        v, v+1, v+2, v+3,
        v+4, v+5, v+6, v+7,
        v+8, v+9, v+10, v+11,
        v+12, v+13, v+14, v+15
    );

    out[i] = r + v;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_out, bytes);

    many_args<<<(n + 255) / 256, 256>>>(d_a, d_out, n);

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}