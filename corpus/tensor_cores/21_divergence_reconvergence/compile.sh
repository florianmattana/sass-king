#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

compile_variant() {
  local variant="$1"
  local name="$2"
  nvcc -arch=sm_120 -DVARIANT="${variant}" -o "build/${name}" 21_divergence_reconvergence.cu
  cuobjdump --dump-sass "build/${name}" > "${name}.sass"
}

compile_variant 2100 21a_predicated_if
compile_variant 2101 21b_uniform_branch
compile_variant 2102 21c_lane_divergent_if
compile_variant 2103 21d_if_else_divergent
compile_variant 2104 21e_nested_divergence
compile_variant 2105 21f_break_divergent_loop
compile_variant 2106 21g_continue_divergent_loop
compile_variant 2107 21h_early_return
compile_variant 2108 21i_barrier_after_divergence
compile_variant 2109 21j_warp_vote_control
compile_variant 2110 21k_select_vs_branch
compile_variant 2111 21l_short_branch_vs_long_branch
compile_variant 2112 21m_divergent_memory
compile_variant 2113 21n_divergent_mma_guard
compile_variant 2114 21o_loop_tripcount_per_lane
compile_variant 2115 21p_bounds_check_epilogue
compile_variant 2116 21q_masked_store_tail
compile_variant 2117 21r_divergent_call_inline_pressure
compile_variant 2118 21s_assert_trap_or_error_path
compile_variant 2119 21t_converged_after_vote
