// sass-king step 12i: extreme register pressure with 32 live accumulators
// Doubles the pressure from 12e. Should force spill.
// Compile: nvcc -arch=sm_120 -maxrregcount=24 -o 12i_32acc 12i_32acc.cu
// Dump:    cuobjdump --dump-sass 12i_32acc

__global__ void acc32(const float* a, float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float s00=0, s01=0, s02=0, s03=0, s04=0, s05=0, s06=0, s07=0;
    float s08=0, s09=0, s10=0, s11=0, s12=0, s13=0, s14=0, s15=0;
    float s16=0, s17=0, s18=0, s19=0, s20=0, s21=0, s22=0, s23=0;
    float s24=0, s25=0, s26=0, s27=0, s28=0, s29=0, s30=0, s31=0;

    float v = a[i];

    for (int k = 0; k < iters; ++k) {
        float x = v + (float)k;
        s00 = fmaf(x, s01, s00);
        s01 = fmaf(x, s02, s01);
        s02 = fmaf(x, s03, s02);
        s03 = fmaf(x, s04, s03);
        s04 = fmaf(x, s05, s04);
        s05 = fmaf(x, s06, s05);
        s06 = fmaf(x, s07, s06);
        s07 = fmaf(x, s08, s07);
        s08 = fmaf(x, s09, s08);
        s09 = fmaf(x, s10, s09);
        s10 = fmaf(x, s11, s10);
        s11 = fmaf(x, s12, s11);
        s12 = fmaf(x, s13, s12);
        s13 = fmaf(x, s14, s13);
        s14 = fmaf(x, s15, s14);
        s15 = fmaf(x, s16, s15);
        s16 = fmaf(x, s17, s16);
        s17 = fmaf(x, s18, s17);
        s18 = fmaf(x, s19, s18);
        s19 = fmaf(x, s20, s19);
        s20 = fmaf(x, s21, s20);
        s21 = fmaf(x, s22, s21);
        s22 = fmaf(x, s23, s22);
        s23 = fmaf(x, s24, s23);
        s24 = fmaf(x, s25, s24);
        s25 = fmaf(x, s26, s25);
        s26 = fmaf(x, s27, s26);
        s27 = fmaf(x, s28, s27);
        s28 = fmaf(x, s29, s28);
        s29 = fmaf(x, s30, s29);
        s30 = fmaf(x, s31, s30);
        s31 = fmaf(x, s00, s31);
    }

    out[i] = s00+s01+s02+s03+s04+s05+s06+s07
          + s08+s09+s10+s11+s12+s13+s14+s15
          + s16+s17+s18+s19+s20+s21+s22+s23
          + s24+s25+s26+s27+s28+s29+s30+s31;
}

int main() {
    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_out;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_out, bytes);

    acc32<<<(n + 255) / 256, 256>>>(d_a, d_out, n, 100);

    cudaFree(d_a);
    cudaFree(d_out);
    return 0;
}