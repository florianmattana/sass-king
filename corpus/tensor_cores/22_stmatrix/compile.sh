#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

compile_variant() {
  local variant="$1"
  local name="$2"
  local arch="${3:-sm_120}"
  nvcc -arch="${arch}" -diag-suppress=177 -DVARIANT="${variant}" -o "build/${name}" 22_stmatrix.cu
  cuobjdump --dump-sass "build/${name}" > "${name}.sass"
}

compile_variant 2200 22a_stmatrix_x1
compile_variant 2201 22b_stmatrix_x2
compile_variant 2202 22c_stmatrix_x4
compile_variant 2203 22d_stmatrix_x1_trans
compile_variant 2204 22e_stmatrix_x2_trans
compile_variant 2205 22f_stmatrix_x4_trans
compile_variant 2206 22g_scalar_sts_fallback
compile_variant 2207 22h_layout_x4
compile_variant 2208 22i_layout_x4_trans
compile_variant 2209 22j_stsm_barrier_visibility
compile_variant 2210 22k_stsm_no_barrier_same_thread
compile_variant 2211 22l_hmma_adjacent_stmatrix
