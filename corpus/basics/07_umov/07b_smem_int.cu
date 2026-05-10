// sass-king step 07b: shared memory with int type
// Delta from kernel 06b: __shared__ int instead of float, same count
// Goal: test if UMOV 0x400 depends on shared element type
// Compile: nvcc -arch=sm_120 -o 07b_smem_int 07b_smem_int.cu
// Dump:    cuobjdump --dump-sass 07b_smem_int

__global__ void smem_int(const int* a, int* c, int n) {
    __shared__ int smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem[tid] = a[i];
        __syncthreads();
        int src = (tid + 1) % 256;
        c[i] = smem[src];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(int);

    int *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    smem_int<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}