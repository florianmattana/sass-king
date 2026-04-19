// sass-king step 09l: __activemask() alone
// Delta: pure query of active mask, no sync or shuffle
// Goal: observe which opcode ptxas emits for __activemask
// Compile: nvcc -arch=sm_120 -o 09l_activemask 09l_activemask.cu
// Dump:    cuobjdump --dump-sass 09l_activemask

__global__ void activemask(unsigned int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mask = __activemask();

    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = mask;
    }
}

int main() {
    const int n = 1024;
    const int bytes_out = (n / 32) * sizeof(unsigned int);

    unsigned int *d_c;
    cudaMalloc(&d_c, bytes_out);

    activemask<<<(n + 255) / 256, 256>>>(d_c);

    cudaFree(d_c);
    return 0;
}