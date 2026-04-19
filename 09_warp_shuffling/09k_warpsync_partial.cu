// sass-king step 09k: __syncwarp with partial mask
// Delta from 09e: mask is not 0xffffffff, forcing potentially a real WARPSYNC instruction
// Goal: observe if WARPSYNC opcode appears when the mask is partial
// Compile: nvcc -arch=sm_120 -o 09k_warpsync_partial 09k_warpsync_partial.cu
// Dump:    cuobjdump --dump-sass 09k_warpsync_partial

__global__ void warpsync_partial(const float* a, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[i];

    // Sync only first half of each warp
    __syncwarp(0x0000ffff);
    val += __shfl_xor_sync(0x0000ffff, val, 8);

    c[i] = val;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    warpsync_partial<<<(n + 255) / 256, 256>>>(d_a, d_c);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}