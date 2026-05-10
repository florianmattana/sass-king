// sass-king step 06e: vector through shared memory, smaller shared size
// Delta from step 06b: shared array size 128 floats (512 bytes) instead of 256 (1024 bytes)
// Goal: test hypothesis that UMOV UR4, 0x400 encodes shared memory size in bytes
// Compile: nvcc -arch=sm_120 -o 06e_vector_smem_half 06e_vector_smem_half.cu
// Dump:    cuobjdump --dump-sass 06e_vector_smem_half

__global__ void vector_smem_half(const float* a, float* c, int n) {
    __shared__ float smem[128];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n && tid < 128) {
        smem[tid] = a[i];
        __syncthreads();
        int src = (tid + 1) % 128;
        c[i] = smem[src];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    vector_smem_half<<<(n + 127) / 128, 128>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}