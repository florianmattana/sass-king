// sass-king step 08f: double4 vectorized load
// Delta from kernel 08a: double4 instead of float4 (256 bits via 4 x 64-bit elements)
// Goal: test if 256-bit load uses LDG.E.ENL2.256 when elements are 64-bit
//       or falls back to LDG.E.128 x 2 or something else
// Compile: nvcc -arch=sm_120 -o 08f_double4 08f_double4.cu
// Dump:    cuobjdump --dump-sass 08f_double4

__global__ void double4_add(const double4* a, const double4* b, double4* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double4 va = a[i];
        double4 vb = b[i];
        double4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[i] = vc;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(double4);

    double4 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    double4_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}