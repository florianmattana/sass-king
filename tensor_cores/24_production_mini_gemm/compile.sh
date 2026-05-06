#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

compile_variant() {
  local variant="$1"
  local name="$2"
  nvcc -arch=compute_120a -code=sm_120a -diag-suppress=177 -DVARIANT="${variant}" -o "build/${name}" 24_production_mini_gemm.cu
  cuobjdump --dump-sass "build/${name}" > "${name}.sass"
}

compile_variant 2400 24a_minimal_hmma_tile
compile_variant 2401 24b_minimal_qmma_e4m3_tile
compile_variant 2402 24c_minimal_qmma_e2m1_tile
compile_variant 2403 24d_minimal_omma_fp4_scaled_tile
compile_variant 2404 24e_cp_async_single_stage
compile_variant 2405 24f_cp_async_double_buffer
compile_variant 2406 24g_ldsm_order_ab
compile_variant 2407 24h_accumulator_chain_depth
compile_variant 2408 24i_epilogue_stg_scalar
compile_variant 2409 24j_epilogue_stmatrix_shared
compile_variant 2410 24k_epilogue_shared_to_global
compile_variant 2411 24l_predicated_bounds_epilogue
compile_variant 2412 24m_tile_loop_preserved
compile_variant 2413 24n_tile_loop_unrolled
compile_variant 2414 24o_divergence_guarded_tile
compile_variant 2415 24p_register_pressure_variant
compile_variant 2416 24q_shared_memory_alignment_variant
compile_variant 2417 24r_sparse_qmma_tile
compile_variant 2418 24s_scaled_sparse_omma_tile
compile_variant 2419 24t_full_audit_checklist_dump
compile_variant 2420 24u_warpgroup_absence_check
compile_variant 2421 24v_uniform_register_path
compile_variant 2422 24w_descriptor_or_address_arithmetic
compile_variant 2423 24x_barrier_arrive_wait_variants
compile_variant 2424 24y_vectorized_global_store_epilogue
compile_variant 2425 24z_split_k_or_multi_cta_reduction_stub
compile_variant 2426 24aa_scale_load_path
compile_variant 2427 24ab_metadata_load_path
compile_variant 2428 24ac_nontrivial_layout_strides
compile_variant 2429 24ad_cold_error_or_assert_path
