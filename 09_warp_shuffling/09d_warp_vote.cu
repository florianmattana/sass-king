// sass-king step 09d: warp vote via __ballot_sync
// Delta from earlier SHFL kernels: introduces VOTE pattern (warp-wide predicate reduction)
// Goal: observe VOTE opcode
// Compile: nvcc -arch=sm_120 -o 09d_warp_vote 09d_warp_vote.cu
// Dump:    cuobjdump --dump-sass 09d_warp_vote

__global__ void warp_vote(const float* a, unsigned int* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[i];

    // Each thread votes on whether its value is positive
    unsigned int mask = __ballot_sync(0xffffffff, val > 0.0f);

    // Lane 0 of each warp writes the mask
    if ((threadIdx.x & 31) == 0) {
        c[blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)] = mask;
    }
}

int main() {
    const int n = 1024;
    const int bytes_in = n * sizeof(float);
    const int bytes_out = (n / 32) * sizeof(unsigned int);

    float *d_a;
    unsigned int *d_c;
    cudaMalloc(&d_a, bytes_in);
    cudaMalloc(&d_c, bytes_out);

    warp_vote<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}