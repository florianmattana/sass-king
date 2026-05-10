#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

compile_variant() {
  local variant="$1"
  local name="$2"
  nvcc -arch=compute_120a -code=sm_120a -diag-suppress=177 -DVARIANT="${variant}" -o "build/${name}" 25_stsm_epilogue.cu
  cuobjdump --dump-sass "build/${name}" > "${name}.sass"
}

compile_variant 2500 25a_stsm_x1_runtime_layout
compile_variant 2501 25b_stsm_x2_runtime_layout
compile_variant 2502 25c_stsm_x4_runtime_layout
compile_variant 2503 25d_stsm_x1_trans_runtime_layout
compile_variant 2504 25e_stsm_x2_trans_runtime_layout
compile_variant 2505 25f_stsm_x4_trans_runtime_layout
compile_variant 2506 25g_hmma_to_stsm_epilogue
compile_variant 2507 25h_qmma_to_stsm_epilogue
compile_variant 2508 25i_stsm_to_stg_storeback
compile_variant 2509 25j_sts_fallback_comparison
compile_variant 2510 25k_predicated_tail_storeback
compile_variant 2511 25l_alignment_offset_storeback
compile_variant 2512 25m_barrier_required_visibility
compile_variant 2513 25n_no_barrier_same_thread_visibility
compile_variant 2514 25o_split_accumulator_storeback
compile_variant 2515 25p_runtime_decode_table

if nvcc -arch=sm_120 -DTEST_UNSUPPORTED_STMATRIX_B8 -c 25_stsm_epilogue.cu -o build/25q_negative_invalid_b8_stsm.o > 25q_negative_invalid_b8_stsm.log 2>&1; then
  echo "Expected unsupported b8 STSM probe to fail for sm_120" >&2
  exit 1
fi

nvcc -arch=compute_120a -code=sm_120a -DTEST_UNSUPPORTED_STMATRIX_B8 -c 25_stsm_epilogue.cu -o build/25q_sm120a_b8_stsm.o > build/25q_sm120a_b8_stsm.compile.log 2>&1
cuobjdump --dump-sass build/25q_sm120a_b8_stsm.o > 25q_sm120a_b8_stsm.sass

compile_variant 2517 25r_full_epilogue_checklist
compile_variant 2518 25s_f32_accumulator_to_f16_pack
compile_variant 2519 25t_f32_accumulator_to_bf16_pack
compile_variant 2520 25u_omma_to_stsm_epilogue
compile_variant 2521 25v_sparse_qmma_to_stsm_epilogue
compile_variant 2522 25w_noncontiguous_global_stride
compile_variant 2523 25x_shared_bank_stride_variants
compile_variant 2524 25y_register_pressure_epilogue
compile_variant 2525 25z_full_runtime_layout_decode
