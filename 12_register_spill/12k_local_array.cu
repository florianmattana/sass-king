// sass-king step 12k: static local array int arr[64], guaranteed local memory allocation
// 64 * 4 bytes = 256 bytes cannot fit in registers, must go to local memory
// Expected SASS: STL and LDL instructions appear
// Compile: nvcc -arch=sm_120 -o 12k_local_array 12k_local_array.cu
// Dump:    cuobjdump --dump-sass 12k_local_array

__global__ void local_array(const int* in, int* out, int n, int pattern) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int arr[64];

    // Initialize the array with a pattern that depends on runtime data
    // (so ptxas cannot optimize it away by computing in registers)
    for (int k = 0; k < 64; ++k) {
        arr[k] = in[i] + k * pattern;
    }

    // Indirect access with runtime index - forces ptxas to keep arr in local memory
    int idx = (in[i] & 0x3f);
    int sum = 0;
    for (int k = 0; k < 32; ++k) {
        sum += arr[(idx + k) & 0x3f];
    }

    out[i] = sum + arr[idx];
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(int);

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    local_array<<<(n + 255) / 256, 256>>>(d_in, d_out, n, 7);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}