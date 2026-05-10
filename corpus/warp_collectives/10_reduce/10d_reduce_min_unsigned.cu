// sass-king step 10d: warp reduction min (unsigned int32)
// Expected SASS: REDUX.MIN.U32 UR, R
// Compile: nvcc -arch=sm_120 -o 10d_reduce_min_unsigned 10d_reduce_min_unsigned.cu
// Dump:    cuobjdump --dump-sass 10d_reduce_min_unsigned

#include <climits>

__global__ void reduce_min_unsigned(const unsigned int* a, unsigned int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int val = (i < n) ? a[i] : UINT_MAX;
    unsigned int res = __reduce_min_sync(0xffffffff, val);
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

    reduce_min_unsigned<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}