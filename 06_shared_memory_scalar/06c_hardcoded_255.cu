// sass-king step 06b: vector through shared memory, modulo by compile-time constant
// Delta from step 06: replace `% blockDim.x` with `% 256`
// Goal: test hypothesis that modulo by a power-of-2 constant compiles to AND
// Compile: nvcc -arch=sm_120 -o 06b_vector_smem_mod256 06b_vector_smem_mod256.cu
// Dump:    cuobjdump --dump-sass 06b_vector_smem_mod256

__global__ void vector_smem_mod256(const float* a, float* c, int n) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem[tid] = a[i];
        __syncthreads();
        int src = (tid + 1) % 255;
        c[i] = smem[src];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    vector_smem_mod256<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}