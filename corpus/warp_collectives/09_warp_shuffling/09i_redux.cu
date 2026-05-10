// sass-king step 09i: REDUX opcode via __reduce_add_sync
// Delta from 09a: use hardware reduction instead of 5 SHFL.BFLY
// Goal: verify REDUX opcode (added in Ampere, should still exist on SM120)
// Note: REDUX only supports 32-bit integer operands, not float
// Compile: nvcc -arch=sm_120 -o 09i_redux 09i_redux.cu
// Dump:    cuobjdump --dump-sass 09i_redux

__global__ void redux(const int* a, int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = a[i];

    // Hardware warp-wide reduction (single SASS instruction expected)
    int sum = __reduce_add_sync(0xffffffff, val);

    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = sum;
    }
}

int main() {
    const int n = 1024;
    const int bytes_in = n * sizeof(int);
    const int bytes_out = (n / 32) * sizeof(int);

    int *d_a, *d_c;
    cudaMalloc(&d_a, bytes_in);
    cudaMalloc(&d_c, bytes_out);

    redux<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}