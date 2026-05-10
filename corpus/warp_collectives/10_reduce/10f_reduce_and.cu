// sass-king step 10f: warp reduction AND (uint32 bitwise)
// Expected SASS: REDUX.AND UR, R
// Compile: nvcc -arch=sm_120 -o 10f_reduce_and 10f_reduce_and.cu
// Dump:    cuobjdump --dump-sass 10f_reduce_and

__global__ void reduce_and(const unsigned int* a, unsigned int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int val = (i < n) ? a[i] : 0xffffffffU;
    unsigned int res = __reduce_and_sync(0xffffffff, val);
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

    reduce_and<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}