// sass-king step 07a: minimal shared memory
// Delta from kernel 06: shared of size 1 float instead of 256
// Goal: test if UMOV 0x400 remains when shared is minimal
// Compile: nvcc -arch=sm_120 -o 07a_smem_1 07a_smem_1.cu
// Dump:    cuobjdump --dump-sass 07a_smem_1

__global__ void smem_1(const float* a, float* c, int n) {
    __shared__ float smem[1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (threadIdx.x == 0) smem[0] = a[i];
        __syncthreads();
        c[i] = smem[0];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    smem_1<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}