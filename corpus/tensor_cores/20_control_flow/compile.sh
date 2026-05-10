#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

compile_scalar() {
  local variant="$1"
  local name="$2"
  nvcc -arch=sm_120 -DVARIANT="${variant}" -o "build/${name}" 20_scalar_control_flow.cu
  cuobjdump --dump-sass "build/${name}" > "${name}.sass"
}

compile_hmma() {
  local variant="$1"
  local name="$2"
  nvcc -arch=sm_120 -DVARIANT="${variant}" -o "build/${name}" 20_hmma_control_flow.cu
  cuobjdump --dump-sass "build/${name}" > "${name}.sass"
}

compile_scalar 200 20a_constant_loop_n4
compile_scalar 201 20b_constant_loop_n16
compile_scalar 202 20c_dynamic_loop
compile_scalar 203 20d_constant_loop_unroll1
compile_scalar 204 20e_nested_scalar_4x2
compile_scalar 205 20f_nested_scalar_8x2
compile_hmma   206 20g_nested_hmma_4x2
compile_hmma   207 20h_nested_hmma_8x2
compile_scalar 208 20i_dynamic_short_if
compile_scalar 209 20j_dynamic_large_if
compile_scalar 210 20k_dynamic_break
compile_scalar 211 20l_dynamic_continue
compile_scalar 212 20m_constant_loop_full_unroll
compile_scalar 213 20n_constant_loop_unroll4
compile_scalar 214 20o_dynamic_loop_unroll4
compile_scalar 215 20p_dynamic_dependency
compile_scalar 216 20q_dynamic_independent
compile_scalar 217 20r_dynamic_volatile_store
compile_scalar 218 20s_template_nested_8x2
compile_scalar 219 20t_template_two_instantiations
compile_scalar 220 20u_nested_unique_stores
compile_scalar 221 20v_nested_identical_body
