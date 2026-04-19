// sass-king step 12h: kernel with a __noinline__ device function, to observe spill at CALL boundary
// Forces ptxas to preserve caller-saved registers around the CALL via STL/LDL
// Expected SASS: STL before CALL + LDL after to restore live values
// Compile: nvcc -arch=sm_120 -maxrregcount=24 -o 12h_call_spill 12h_call_spill.cu
// Dump:    cuobjdump --dump-sass 12h_call_spill

__noinline__ __device__ float heavy_function(float a, float b, float c) {
    // Moderate computation to prevent trivial inlining decisions
    float x = a * b + c;
    float y = b * c + a;
    float z = c * a + b;
    return x * y + z;
}

__global__ void call_with_live_state(
    const float* a, const float* b, const float* c, float* out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Build up a lot of live state before the CALL
    float v0 = a[i], v1 = b[i], v2 = c[i];
    float v3 = a[i] + 1.0f, v4 = b[i] + 2.0f, v5 = c[i] + 3.0f;
    float v6 = v0 * v1, v7 = v2 * v3, v8 = v4 * v5;

    // Function call that cannot be inlined; ptxas must preserve v0..v8
    float r = heavy_function(v0, v1, v2);

    // Use all the pre-call values after the CALL returns
    // This forces ptxas to keep them live across the call boundary
    out[i] = r + v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_out, bytes);

    call_with_live_state<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, d_out, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);
    return 0;
}