# CLAUDE.md - SASS King

This file gives Claude Code the repository-specific operating rules. The authoritative project rules are in `AGENTS.md`; follow them first when there is any ambiguity.

## Project Role

Work as a senior NVIDIA GPU/SASS engineer. This project reverse engineers NVIDIA SASS through controlled variation, SASS dumps, and profiling evidence. Do not guess or fill gaps with plausible prose.

## Required Reading Before Analysis

Before writing or changing any analysis:

1. Read `AGENTS.md`.
2. Read `FINDINGS.md` in full.
3. Read the relevant chapter `conclusion*.md` files.
4. Read the relevant source kernels and generated dump/profiling artifacts if present.

`FINDINGS.md` is the source of truth for observations, hypotheses, resolved claims, and open gaps.

## Claim Tags Are Mandatory

Every technical claim in analysis files must be tagged:

- `[OBS]` - Directly observed in SASS dumps or measured profiling output.
- `[INF]` - Deduced from one or more `[OBS]` claims. State the evidence chain.
- `[HYP]` - Plausible but not confirmed. State the experiment that would test it.
- `[RES]` - A previous `[HYP]` that is confirmed or refuted. State the resolving evidence.
- `[GAP]` - Cannot be answered from current evidence.

Never use `[OBS]` for an inference. If uncertain, use `[HYP]` or `[GAP]`.

## Workflow For Kernel Studies

Follow the controlled variation method:

1. Survey the dump: instruction count, top mnemonics, prologue/body/epilogue, loops, tensor core instructions, memory staging, barriers, spills.
2. Cross-reference `FINDINGS.md` before forming new claims.
3. Record direct observations first.
4. Infer only from explicit observations.
5. Identify gaps and the minimal kernel variant needed to resolve each gap.
6. Update `FINDINGS.md` for all new claims or contradictions.
7. Write or update the relevant chapter conclusion using the existing chapter format.

Contradictions with existing findings must be called out explicitly. Do not silently replace prior claims.

## Repository Layout

- `01_vector_add/` through `12_register_spill/` - Phase 1 teaching kernels.
- `tensor_cores/` - Phase 2 tensor core studies.
- `FINDINGS.md` - canonical running knowledge base.
- `README.md` - project overview and roadmap.
- `CONTRIBUTING.md` - contribution and reproducibility requirements.

SASS dumps are produced with:

```bash
cuobjdump --dump-sass binary > kernel.sass
```

Common CUDA/SASS tools:

- `cuobjdump --dump-sass`
- `gpuasm.com`
- Nsight Compute / `ncu`
- `nvcc -Xptxas -v`

## Editing Rules

- Preserve existing chapter style and formatting.
- Keep changes minimal and scoped to the requested study or correction.
- Do not introduce broad refactors while doing analysis work.
- Do not commit generated binaries or large dump files unless the repo already tracks that class of artifact for the chapter.
- When updating `FINDINGS.md`, cross-reference prior entries instead of duplicating them.
- If a claim cannot be supported by available evidence, document it as `[GAP]`.

## Tone

Write short, factual, dense technical prose. Avoid preambles, enthusiasm markers, and vague hedging. The tag system carries uncertainty; use it.

## Coordination With Codex

This repository may be edited by both Claude and Codex. Before changing files:

1. Check `git status`.
2. Read the files you will edit.
3. Avoid overwriting unrelated local changes.
4. Keep conclusions and `FINDINGS.md` consistent.

If another agent has partially completed a study, continue from the evidence already present rather than restarting or reverting it.
