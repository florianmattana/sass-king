// sass-king step 06h: integer division by runtime blockDim.x
// Delta from step 06: replace modulo with division
// Goal: observe if div uses a separate slowpath function, or reuses __cuda_sm20_rem_u16
// Compile: nvcc -arch=sm_120 -o 06h_hardcoded_div 06h_hardcoded_div.cu
// Dump:    cuobjdump --dump-sass 06h_hardcoded_div

__global__ void vector_smem_div(const float* a, float* c, int n) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem[tid] = a[i];
        __syncthreads();
        int src = (tid + 1) / blockDim.x;
        c[i] = smem[src];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    vector_smem_div<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}