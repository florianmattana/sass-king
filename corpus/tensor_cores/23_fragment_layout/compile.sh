#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

compile_variant() {
  local variant="$1"
  local name="$2"
  nvcc -arch=compute_120a -code=sm_120a -diag-suppress=177 -DVARIANT="${variant}" -o "build/${name}" 23_fragment_layout.cu
  cuobjdump --dump-sass "build/${name}" > "${name}.sass"
}

compile_variant 2300 23a_e2m1_pack_baseline
compile_variant 2301 23b_e3m2_pack_baseline
compile_variant 2302 23c_e2m3_pack_baseline
compile_variant 2303 23d_mixed_e3m2_e2m3
compile_variant 2304 23e_mixed_e2m3_e3m2
compile_variant 2305 23f_e2m1_lane_probe
compile_variant 2306 23g_e3m2_lane_probe
compile_variant 2307 23h_e2m3_lane_probe
compile_variant 2308 23i_scale_factor_interaction
compile_variant 2309 23j_ldmatrix_to_qmma_path
compile_variant 2310 23k_direct_register_qmma_path
compile_variant 2311 23l_runtime_decode_probe
compile_variant 2313 23n_cross_reference_ch14_ch15
compile_variant 2314 23o_unsigned_vs_signed_interpretation
compile_variant 2315 23p_scale_vector_layout
compile_variant 2316 23q_metadata_independence_check
compile_variant 2317 23r_operand_order_layout
compile_variant 2318 23s_k_tile_boundary_probe
compile_variant 2319 23t_register_pair_boundary_probe
compile_variant 2320 23u_zero_nan_inf_special_values
compile_variant 2321 23v_shared_memory_alignment_probe

if nvcc -arch=compute_120a -code=sm_120a -diag-suppress=177 -DTEST_INVALID_FORMAT -DVARIANT=2300 \
  -o build/23m_negative_invalid_format_probe 23_fragment_layout.cu > 23m_negative_invalid_format_probe.log 2>&1; then
  echo "negative invalid-format probe unexpectedly compiled" >> 23m_negative_invalid_format_probe.log
  exit 1
fi
