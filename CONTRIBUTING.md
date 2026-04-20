# Contributing to SASS King

SASS King is a systematic reverse engineering of NVIDIA SASS across GPU architectures. Contributions are welcome. Read this document before opening an issue or a PR.

## Two types of contributions

### Type A: SASS dumps for architectures not owned by the maintainer

If you have an A100, A6000, H100, 4090, or any GPU other than an RTX 5070 Ti, you can contribute a dump of an existing kernel in the project. The analysis itself is done by the maintainer. You keep contributor credit on the resulting study.

What you submit:

1. The raw `cuobjdump --dump-sass` output of the kernel compiled for your GPU's SM.
2. The full `nvcc` compile command used.
3. GPU model, CUDA toolkit version, driver version.
4. Commit hash of the kernel source you compiled.

Open an issue using the `SASS dump contribution` template **before** doing the work. The maintainer will confirm the kernel choice and compilation flags to keep cross-arch dumps comparable.

### Type B: new kernel studies, corrections, or analysis

If you want to propose a new kernel study, correct an existing finding, or add analysis on an existing dump, this is a heavier contribution and requires adherence to the methodology.

Open an issue using the `Kernel study proposal` or `Correction` template first. PRs without a prior approved issue will be closed.

## Reproducibility requirements

Every dump must include:

* GPU model and SM version.
* CUDA toolkit version and exact driver version.
* Full `nvcc` command with all flags used to produce the cubin.
* Commit hash of the source file compiled.
* Raw `cuobjdump --dump-sass` output. No editing, no truncation.
* NCU report

Dumps without this metadata cannot be accepted. They are not reproducible and therefore not usable as reference.

## Hypothesis versus verified discipline

Every claim in a PR must be tagged as one of:

* **Observed.** Directly visible in the SASS or measured in NCU. State the line or the metric.
* **Inferred.** Not directly visible but deduced from observed evidence. State the evidence chain.
* **Hypothesis.** Not yet verified. State what microbenchmark or variation would confirm or reject it.

Mixing the three is the central failure mode of SASS analysis. Dumps without this discipline are indistinguishable from guessing and will be rejected.

## Naming conventions

Kernel studies follow `NN_short_name/` at the repo root, with `NN` incremented sequentially. Inside each kernel directory:

```
NN_short_name/
  source.cu              source code compiled
  compile.sh             compilation command
  sm_120.sass            dump from RTX 5070 Ti (default)
  dumps/
    sm_80.sass           dump from A100 (optional, contributor)
    sm_89.sass           dump from 4090 (optional, contributor)
    sm_90a.sass          dump from H100 (optional, contributor)
    sm_XX.meta.md        metadata for each cross-arch dump
  conclusions.md         main analysis
```

Each file under `dumps/sm_XX.sass` is paired with a `dumps/sm_XX.meta.md` giving GPU, driver, CUDA version, and compile command.

Branch naming:

* `kernel-NN-description` for new kernel studies.
* `dump-smXX-kernel-NN` for cross-arch dumps.
* `fix-description` for corrections.

## Style

Technical content is written in English. Conclusions follow the structure of existing chapter conclusions: source, SASS section by section, observations, hypotheses, open questions. Cross-reference `FINDINGS.md`, `CONTROL_CODES.md`, and `OPCODE_MODIFIERS.md` when discussing patterns already covered there.

## PR workflow

1. Open an issue first using the correct template. Wait for a go-ahead before starting work.
2. Fork the repo. Create a branch using the naming convention above.
3. Make the minimum changes needed. One kernel study per PR. One dump per PR. One correction per PR.
4. Fill the PR template completely. Incomplete templates are not reviewed.
5. The maintainer reviews. Expect questions about reproducibility and about the observed / inferred / hypothesis split.
6. Merge happens after approval.

## What is out of scope

General GPU optimization advice without SASS-level evidence. Benchmarks of kernel speedups without corresponding dump analysis. Cross-posting content from NVIDIA documentation without added observation. Opinions on architectural choices without measured data.

## License

By contributing, you agree that your contribution is licensed under the project license.
