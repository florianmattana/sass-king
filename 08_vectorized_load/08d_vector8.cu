// sass-king step 08d: 8 floats per thread via struct
// Delta from kernel 08a: 8-float struct instead of float4
// Goal: test if LDG.E.256 exists, or if ptxas splits into 2 x LDG.E.128
// Compile: nvcc -arch=sm_120 -o 08d_vector8 08d_vector8.cu
// Dump:    cuobjdump --dump-sass 08d_vector8

struct __align__(32) float8 {
    float x0, x1, x2, x3, x4, x5, x6, x7;
};

__global__ void vector8_add(const float8* a, const float8* b, float8* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float8 va = a[i];
        float8 vb = b[i];
        float8 vc;
        vc.x0 = va.x0 + vb.x0;
        vc.x1 = va.x1 + vb.x1;
        vc.x2 = va.x2 + vb.x2;
        vc.x3 = va.x3 + vb.x3;
        vc.x4 = va.x4 + vb.x4;
        vc.x5 = va.x5 + vb.x5;
        vc.x6 = va.x6 + vb.x6;
        vc.x7 = va.x7 + vb.x7;
        c[i] = vc;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float8);

    float8 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vector8_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}