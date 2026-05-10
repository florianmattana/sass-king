// sass-king step 10b: warp reduction min (signed int32)
// Expected SASS: REDUX.MIN.S32 UR, R
// Compile: nvcc -arch=sm_120 -o 10b_reduce_min_signed 10b_reduce_min_signed.cu
// Dump:    cuobjdump --dump-sass 10b_reduce_min_signed

#include <climits>

__global__ void reduce_min_signed(const int* a, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (i < n) ? a[i] : INT_MAX;
    int res = __reduce_min_sync(0xffffffff, val);
    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = res;
    }
}

int main() {
    const int n = 1024;
    const int bytes_a = n * sizeof(int);
    const int bytes_c = (n / 32) * sizeof(int);

    int *d_a, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_c, bytes_c);

    reduce_min_signed<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}