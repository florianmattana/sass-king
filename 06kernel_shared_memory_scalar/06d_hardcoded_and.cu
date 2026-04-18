// sass-king step 06d: vector through shared memory, bitwise AND
// Delta from step 06b: replace `% 256` with `& 255`
// Goal: test hypothesis that `& 255` and `% 256` produce identical SASS
// Compile: nvcc -arch=sm_120 -o 06d_hardcoded_and 06d_hardcoded_and.cu
// Dump:    cuobjdump --dump-sass 06d_hardcoded_and

__global__ void vector_smem_and255(const float* a, float* c, int n) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem[tid] = a[i];
        __syncthreads();
        int src = (tid + 1) & 255;
        c[i] = smem[src];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    vector_smem_and255<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}