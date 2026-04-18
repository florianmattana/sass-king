// sass-king step 06: shared memory scalar staging
// Delta: introduce __shared__ buffer between load and store
// Targets: STS, LDS, BAR.SYNC
// Compile: nvcc -arch=sm_120 -o 06_shared_scalar 06_shared_scalar.cu
// sass-king step 06: vector through shared memory
// Delta from step 01: data transits through __shared__
// Goal: observe LDS, STS, BAR.SYNC
// Compile: nvcc -arch=sm_120 -o 06_vector_smem 06_vector_smem.cu
// Dump:    cuobjdump --dump-sass 06_vector_smem

__global__ void vector_smem(const float* a, float* c, int n) {
    __shared__ float smem[256];
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

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    vector_smem<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}