# Knowledge Base

This directory contains the project-wide outputs produced from the chapter probes.

| File | Purpose |
|---|---|
| `FINDINGS.md` | Running source of truth for observations, hypotheses, resolutions, gaps, cross-chapter invariants, and canonical patterns. |
| `ISA_COVERAGE.md` | Completeness tracker for the long-term ISA documentation effort. |
| `SASS_INSTRUCTIONS_SM120.md` | Evidence-backed SM120 / SM120a instruction inventory. |
| `DENVDIS_INTEGRATION.md` | Plan and results table for validating redplait/denvdis as the bit-level SM120 cross-check backend. |
| `encoding/schema.md` | Shared format for instruction-family encoding pages. |
| `encoding/STSM.md` | Pilot encoding page for matrix-store instructions. |
| `encoding/LDSM.md` | Pilot encoding page for matrix-load instructions. |
| `encoding/QMMA.md` | Pilot encoding page for Blackwell dense/sparse FP8/FP6/FP4 MMA instructions. |

Chapter-local files stay with their kernels. For example, `corpus/tensor_cores/25_stsm_epilogue/conclusion25.md` documents the local experiment, while `knowledge/FINDINGS.md` records the project-wide result.
