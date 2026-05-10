// sass-king step 09j: MATCH opcode
// Delta: introduces __match_*_sync (added in Volta)
// Goal: observe MATCH.ANY and MATCH.ALL opcodes
// Compile: nvcc -arch=sm_120 -o 09j_match 09j_match.cu
// Dump:    cuobjdump --dump-sass 09j_match

__global__ void match_test(const int* a, unsigned int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = a[i];

    // MATCH.ANY: mask of lanes whose val matches mine
    unsigned int match_any = __match_any_sync(0xffffffff, val);

    // MATCH.ALL: check if all lanes have the same val
    int pred;
    unsigned int match_all = __match_all_sync(0xffffffff, val, &pred);

    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = match_any ^ match_all;
    }
}

int main() {
    const int n = 1024;
    const int bytes_in = n * sizeof(int);
    const int bytes_out = (n / 32) * sizeof(unsigned int);

    int *d_a;
    unsigned int *d_c;
    cudaMalloc(&d_a, bytes_in);
    cudaMalloc(&d_c, bytes_out);

    match_test<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}