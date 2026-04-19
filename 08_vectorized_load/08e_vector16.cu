// sass-king step 08e: 16 floats per thread via struct
// Delta from kernel 08d: 16-float struct instead of 8
// Goal: test if LDG.E.ENL4.512 exists, or if ptxas splits into 2x LDG.E.ENL2.256
// Compile: nvcc -arch=sm_120 -o 08e_vector16 08e_vector16.cu
// Dump:    cuobjdump --dump-sass 08e_vector16

struct __align__(64) float16 {
    float x0, x1, x2, x3, x4, x5, x6, x7;
    float x8, x9, xa, xb, xc, xd, xe, xf;
};

__global__ void vector16_add(const float16* a, const float16* b, float16* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float16 va = a[i];
        float16 vb = b[i];
        float16 vc;
        vc.x0 = va.x0 + vb.x0;
        vc.x1 = va.x1 + vb.x1;
        vc.x2 = va.x2 + vb.x2;
        vc.x3 = va.x3 + vb.x3;
        vc.x4 = va.x4 + vb.x4;
        vc.x5 = va.x5 + vb.x5;
        vc.x6 = va.x6 + vb.x6;
        vc.x7 = va.x7 + vb.x7;
        vc.x8 = va.x8 + vb.x8;
        vc.x9 = va.x9 + vb.x9;
        vc.xa = va.xa + vb.xa;
        vc.xb = va.xb + vb.xb;
        vc.xc = va.xc + vb.xc;
        vc.xd = va.xd + vb.xd;
        vc.xe = va.xe + vb.xe;
        vc.xf = va.xf + vb.xf;
        c[i] = vc;
    }
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float16);

    float16 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    vector16_add<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}