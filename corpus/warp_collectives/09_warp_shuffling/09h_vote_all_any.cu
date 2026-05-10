// sass-king step 09h: VOTE.ALL and VOTE.ANY distinction
// Delta from 09d: test both __all_sync and __any_sync in same kernel
// Goal: check if VOTE.ALL is a distinct opcode or if all votes funnel through VOTE.ANY
// Compile: nvcc -arch=sm_120 -o 09h_vote_all_any 09h_vote_all_any.cu
// Dump:    cuobjdump --dump-sass 09h_vote_all_any

__global__ void vote_all_any(const float* a, int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[i];

    int any_result = __any_sync(0xffffffff, val > 0.0f);
    int all_result = __all_sync(0xffffffff, val > 0.0f);

    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = any_result + (all_result << 1);
    }
}

int main() {
    const int n = 1024;
    const int bytes_in = n * sizeof(float);
    const int bytes_out = (n / 32) * sizeof(int);

    float *d_a;
    int *d_c;
    cudaMalloc(&d_a, bytes_in);
    cudaMalloc(&d_c, bytes_out);

    vote_all_any<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}