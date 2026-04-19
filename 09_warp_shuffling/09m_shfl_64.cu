// sass-king step 09m: SHFL on 64-bit operand
// Delta from 09c: double value instead of float
// Goal: observe whether 64-bit SHFL is 1 instruction or split into 2
// Compile: nvcc -arch=sm_120 -o 09m_shfl_64 09m_shfl_64.cu
// Dump:    cuobjdump --dump-sass 09m_shfl_64

__global__ void shfl_64(const double* a, double* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double val = a[i];

    // Broadcast lane 0's 64-bit value to all
    double broadcast = __shfl_sync(0xffffffff, val, 0);

    c[i] = val + broadcast;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(double);

    double *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    shfl_64<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}