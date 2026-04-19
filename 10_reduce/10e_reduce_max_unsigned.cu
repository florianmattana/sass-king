// sass-king step 10e: warp reduction max (unsigned int32)
// Expected SASS: REDUX.MAX.U32 UR, R
// Compile: nvcc -arch=sm_120 -o 10e_reduce_max_unsigned 10e_reduce_max_unsigned.cu
// Dump:    cuobjdump --dump-sass 10e_reduce_max_unsigned

__global__ void reduce_max_unsigned(const unsigned int* a, unsigned int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int val = (i < n) ? a[i] : 0;
    unsigned int res = __reduce_max_sync(0xffffffff, val);
    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = res;
    }
}

int main() {
    const int n = 1024;
    const int bytes_a = n * sizeof(unsigned int);
    const int bytes_c = (n / 32) * sizeof(unsigned int);

    unsigned int *d_a, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_c, bytes_c);

    reduce_max_unsigned<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}