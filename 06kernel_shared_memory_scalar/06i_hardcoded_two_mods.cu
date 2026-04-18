// sass-king step 06i: two modulos with different runtime divisors
// Delta from step 06: one modulo by blockDim.x, one by blockDim.y (two distinct runtime values)
// Goal: see if ptxas generates one shared helper or two distinct helpers
// Compile: nvcc -arch=sm_120 -o 06i_hardcoded_two_mods 06i_hardcoded_two_mods.cu
// Dump:    cuobjdump --dump-sass 06i_hardcoded_two_mods

__global__ void vector_two_mods(const float* a, float* c, int n) {
    __shared__ float smem[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (i < n) {
        smem[tid] = a[i];
        __syncthreads();
        int src1 = (tid + 1) % blockDim.x;
        int src2 = (tid + 2) % blockDim.y;
        c[i] = smem[src1] + smem[src2];
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    dim3 block(256, 1, 1);
    dim3 grid((n + 255) / 256, 1, 1);
    vector_two_mods<<<grid, block>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}