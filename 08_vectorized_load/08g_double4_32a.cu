// sass-king step 08g: double4 with explicit 32-byte alignment
// Delta from kernel 08f: double4_32a instead of deprecated double4 (which has 16-byte alignment)
// Goal: test if 256-bit load becomes possible when alignment is guaranteed
// Compile: nvcc -arch=sm_120 -o 08g_double4_32a 08g_double4_32a.cu
// Dump:    cuobjdump --dump-sass 08g_double4_32a

__global__ void double4_32a_add(const double4_32a* a, const double4_32a* b, double4_32a* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double4_32a va = a[i];
        double4_32a vb = b[i];
        double4_32a vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[i] = vc;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(double4_32a);

    double4_32a *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    double4_32a_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}