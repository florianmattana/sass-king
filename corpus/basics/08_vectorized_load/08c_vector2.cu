// sass-king step 08c: float2 vectorized load
// Delta from kernel 08a: float2 instead of float4
// Goal: test if LDG.E.64 exists and how ptxas handles 2-element vectors
// Compile: nvcc -arch=sm_120 -o 08c_vector2 08c_vector2.cu
// Dump:    cuobjdump --dump-sass 08c_vector2

__global__ void vector2_add(const float2* a, const float2* b, float2* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float2 va = a[i];
        float2 vb = b[i];
        float2 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        c[i] = vc;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float2);

    float2 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vector2_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}