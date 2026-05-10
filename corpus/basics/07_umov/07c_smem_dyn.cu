// sass-king step 07c: dynamic shared memory
// Delta from kernel 06b: extern __shared__ instead of fixed size array
// Goal: test what UMOV holds when shared size is unknown at compile time
// Compile: nvcc -arch=sm_120 -o 07c_smem_dyn 07c_smem_dyn.cu
// Dump:    cuobjdump --dump-sass 07c_smem_dyn

extern __shared__ float smem[];

__global__ void smem_dyn(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem[tid] = a[i];
        __syncthreads();
        int src = (tid + 1) % blockDim.x;
        c[i] = smem[src];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);
    const int smem_bytes = 256 * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    smem_dyn<<<(n + 255) / 256, 256, smem_bytes>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}