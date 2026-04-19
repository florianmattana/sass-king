// sass-king step 11b: integer division by runtime variable (u64)
// Expected SASS: CALL to a 64-bit division helper, longer than u32 version
// Compile: nvcc -arch=sm_120 -o 11b_div_u64_runtime 11b_div_u64_runtime.cu
// Dump:    cuobjdump --dump-sass 11b_div_u64_runtime

__global__ void div_u64_runtime(const unsigned long long* a, unsigned long long* c, int n, unsigned long long d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] / d;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(unsigned long long);

    unsigned long long *d_a, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    div_u64_runtime<<<(n + 255) / 256, 256>>>(d_a, d_c, n, 7ULL);

    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}