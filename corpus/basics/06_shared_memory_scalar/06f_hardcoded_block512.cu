// sass-king step 06f: vector through shared memory, block size 512
// Delta from step 06b: launch with 512 threads/block instead of 256
// Goal: test if UMOV 0x400 encodes block size or is architectural constant
// Compile: nvcc -arch=sm_120 -o 06f_hardcoded_block512 06f_hardcoded_block512.cu
// Dump:    cuobjdump --dump-sass 06f_hardcoded_block512

__global__ void vector_smem_block512(const float* a, float* c, int n) {
    __shared__ float smem[512];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem[tid] = a[i];
        __syncthreads();
        int src = (tid + 1) % 512;
        c[i] = smem[src];
    }
}

int main() {
    const int n = 2048;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    vector_smem_block512<<<(n + 511) / 512, 512>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}   