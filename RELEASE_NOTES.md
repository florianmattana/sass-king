# Release Notes

## v0.1.0-research-corpus - draft

This release marks SASS King as a readable research corpus and evidence-backed knowledge base for NVIDIA SASS. It does not ship a standalone tool.

## Scope

Included in the v0.1 boundary:

- controlled CUDA kernel corpus through chapters 01-25;
- reorganized `corpus/` layout with section indexes;
- project-wide `knowledge/FINDINGS.md` with navigation index and claim discipline;
- SM120 / SM120a instruction glossary;
- pilot encoding pages for `LDSM`, `STSM`, `QMMA`, and partial control-code modeling;
- denvdis integration notes and representative cross-validation results;
- contribution rules for evidence tagging and dump metadata.

Not included yet:

- standalone SASS disassembler;
- standalone SASS assembler;
- Ghidra plugin;
- one-command audit CLI;
- completed cross-architecture replay;
- production-library audit suite.

## Reproducible today

The executable part of v0.1 is the controlled CUDA corpus. A reader can compile a kernel, dump NVIDIA SASS with `cuobjdump`, and compare the result with the chapter conclusion.

Example:

```bash
cd corpus/basics/01_vector_add
nvcc -arch=sm_120 kernel1.cu -o vector_add
cuobjdump --dump-sass vector_add > sm_120.sass
```

## Main artifacts

| Artifact | Purpose |
|---|---|
| `README.md` | Project overview, roadmap, and related work. |
| `docs/START_HERE.md` | Minimal onboarding path. |
| `corpus/README.md` | Corpus section map and reproduction model. |
| `knowledge/FINDINGS.md` | Primary source of truth for observations, hypotheses, resolutions, and gaps. |
| `knowledge/SASS_INSTRUCTIONS_SM120.md` | Evidence-backed instruction-family inventory. |
| `knowledge/encoding/` | Reusable instruction-family notes. |
| `knowledge/DENVDIS_INTEGRATION.md` | denvdis validation status and policy. |

## Release checklist

- [ ] Verify local Markdown links.
- [ ] Confirm the release file set contains only intended project content.
- [ ] Exclude non-project workflow files from the release.
- [ ] Tag after the release branch is merged.
