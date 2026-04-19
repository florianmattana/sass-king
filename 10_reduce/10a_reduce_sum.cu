// sass-king step 10a: warp reduction sum (int32)
// Expected SASS: REDUX.SUM.S32 UR, R
// Compile: nvcc -arch=sm_120 -o 10a_reduce_sum 10a_reduce_sum.cu
// Dump:    cuobjdump --dump-sass 10a_reduce_sum

__global__ void reduce_sum(const int* a, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (i < n) ? a[i] : 0;
    int res = __reduce_add_sync(0xffffffff, val);
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

    reduce_sum<<<(n + 255) / 256, 256>>>(d_a, d_c, n);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}