# Encoding Page Schema

Encoding pages are structured notes for instruction-family matching. They are not complete hardware specifications unless explicitly marked as such.

## Required Sections

Each page should use this structure:

1. `# <Family>`
2. `## Scope`
3. `## Evidence`
4. `## Canonical Forms`
5. `## Operand Model`
6. `## Modifier / Field Model`
7. `## Matching Notes`
8. `## Open Gaps`

## Claim Tags

* [OBS] Directly observed in a local SASS dump, ptxas log, runtime output, or measured profile.
* [INF] Inferred from controlled comparisons across observed variants.
* [HYP] Plausible but not validated.
* [RES] Resolved claim from prior observations or hypotheses.
* [GAP] Missing or incomplete evidence.

## Canonical Form Table

Use this table shape:

| Key | Distilled form | Meaning | Why it matters | Evidence | Notes |
|---|---|---|---|---|---|
| `FAMILY_OPERANDS_MODIFIERS` | `MNEMONIC operands` | Short literal description. | Short audit or matching relevance. | `path/to/file.sass` | [OBS] Short note. |

## Field Model Table

Use this table shape:

| Field | Values observed | Evidence | Status |
|---|---|---|---|
| `shape` | `M88`, `MT88` | chapter paths | [OBS] Meaning. |

## Rules

* Do not include raw bit grids unless the bit placement is backed by local evidence or explicitly marked [GAP].
* Prefer normalized keys that can become matcher signatures.
* Add short literal explanations to tables that introduce an instruction, operand, field, or canonical form. The goal is glossary value, not long prose inside a table.
* Separate target-specific claims (`sm_120` vs `sm_120a`).
* Keep chapter-local narrative in `corpus/tensor_cores/*/conclusion*.md`; keep reusable matching facts here.
