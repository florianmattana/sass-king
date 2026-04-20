// sass-king step 06g: vector through two shared memory buffers
// Delta from step 06b: two __shared__ arrays instead of one
// Goal: observe how ptxas handles multiple shared buffers (single UMOV + offsets, or two UMOV?)
// Compile: nvcc -arch=sm_120 -o 06g_hardcoded_two_smem 06g_hardcoded_two_smem.cu
// Dump:    cuobjdump --dump-sass 06g_hardcoded_two_smem

__global__ void vector_two_smem(const float* a, float* c, int n) {
    __shared__ float smem_a[256];
    __shared__ float smem_b[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem_a[tid] = a[i];
        smem_b[tid] = a[i] + 1.0f;
        __syncthreads();
        int src = (tid + 1) % 256;
        c[i] = smem_a[src] + smem_b[src];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    vector_two_smem<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}