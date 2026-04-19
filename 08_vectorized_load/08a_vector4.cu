// sass-king step 08a: natural vectorized load (float4)
// Delta from kernel 01: each thread processes 1 float4 instead of 1 float
// Goal: observe LDG.E.128 and how ptxas handles vector types
// Compile: nvcc -arch=sm_120 -o 08a_vector4 08a_vector4.cu
// Dump:    cuobjdump --dump-sass 08a_vector4

__global__ void vector4_add(const float4* a, const float4* b, float4* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float4 va = a[i];
        float4 vb = b[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[i] = vc;
    }
}

int main() {
    const int n = 1024;  // n float4 = 4096 floats
    const int bytes = n * sizeof(float4);

    float4 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vector4_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}