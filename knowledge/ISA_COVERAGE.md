# ISA Coverage Tracker

This tracker measures coverage against a historical SASS instruction-family seed list. It is not proof that every listed family exists on SM120.

Claims use the project tags:

* [OBS] Observed in local SASS dumps, logs, or project findings.
* [INF] Inferred from local project structure or controlled comparisons.
* [HYP] Plausible but not validated locally.
* [GAP] Not yet observed, not yet cataloged, or not yet tested.

## Purpose

SASS King aims to document the full ISA over time, but not every family needs the same depth immediately. [INF]

Coverage has three levels:

| Level | Meaning | Expected output |
|---|---|---|
| Inventory | The family is listed in the coverage tracker, even if not observed locally. | Row in this file. |
| Glossary | The family has a short explanation, source trigger, and importance. | Row in `SASS_INSTRUCTIONS_SM120.md`. |
| Encoding | The family has canonical forms, operands, modifiers, and matching notes. | Page under `encoding/`. |

## Current Counts

| Metric | Count | Status |
|---|---:|---|
| Historical seed families including removed and reverse-engineered extras | 293 | [INF] Counted from the seed list in this file. |
| Seed families remaining after applying the listed removals | 256 | [INF] Removal arithmetic only; this is not proof of SM120 support. |
| Families extracted from local `.sass` dumps by mnemonic scan | 62 | [INF] Raw extraction includes real observations but still needs manual review. |
| Families with glossary rows in `SASS_INSTRUCTIONS_SM120.md` | 31 | [INF] Current documented glossary scope. |
| Families with dedicated encoding pages | 3 | [OBS] `LDSM`, `STSM`, `QMMA`. |

## Status Values

| Status | Meaning |
|---|---|
| `documented` | Observed locally and explained in `SASS_INSTRUCTIONS_SM120.md`. |
| `observed_uncataloged` | Observed locally, but not yet promoted into the glossary inventory. |
| `watchlist` | Listed in the historical seed, but not yet observed locally. |
| `removed_or_out_of_scope` | Listed as removed before Blackwell or explicitly out of SM120 scope. |
| `needs_probe` | Important enough to justify a controlled kernel or production audit search. |
| `encoding_page` | Has a dedicated page under `knowledge/encoding/`. |

## SM120 / SM120a Documented Families

These families are already represented in `SASS_INSTRUCTIONS_SM120.md`. [OBS]

| Family | Coverage | Detailed page | Notes |
|---|---|---|---|
| HMMA | documented |  | Baseline FP16/BF16 MMA family. |
| QMMA | documented | `encoding/QMMA.md` | Dense, scaled, and sparse low-precision MMA family. |
| OMMA | documented |  | FP4 peak MMA family observed through `.SF` forms. |
| LDSM | documented | `encoding/LDSM.md` | Matrix-fragment shared-memory load family. |
| STSM | documented | `encoding/STSM.md` | Matrix-fragment shared-memory store family. |
| LDGSTS | documented |  | Async global-to-shared staging path. |
| LDGDEPBAR | documented |  | Async-copy dependency helper. |
| DEPBAR | documented |  | Async-copy wait path. |
| BAR | documented |  | CTA barrier form. |
| LDS | documented |  | Shared-memory load family. |
| STS | documented |  | Shared-memory store family. |
| LDG | documented |  | Global-memory load family. |
| STG | documented |  | Global-memory store family. |
| REDG | documented |  | Global reduction writeback observed in split-K style stub. |
| BRA | documented |  | Branch family. |
| EXIT | documented |  | Kernel exit form. |
| BSSY | documented |  | Reconvergence marker. |
| BSYNC | documented |  | Reconvergence marker. |
| WARPSYNC | documented |  | Warp synchronization form. |
| SHFL | documented |  | Warp shuffle family. |
| VOTE | documented |  | Warp vote family. |
| REDUX | documented |  | Warp reduction family. |
| S2UR | documented |  | Special-register to uniform-register path. |
| R2UR | documented |  | Register to uniform-register path. |
| UMOV | documented |  | Uniform move family. |
| ULEA | documented |  | Uniform address arithmetic. |
| LDCU | documented |  | Uniform constant load family. |

## Observed But Not Yet Cataloged

These mnemonics appear in local dumps or `knowledge/FINDINGS.md`, but they still need glossary rows or a decision to keep them as low-priority background families. [OBS]

| Family | Evidence | Priority | Next action |
|---|---|---|---|
| CS2R | Chapters 11 and tensor-core latency probes | medium | Add special-register read glossary entry. |
| CS2UR | Chapters 13-17, 19 and findings | high | Add uniform special-register read glossary entry. |
| BPT | Local dump extraction | low | Verify context before cataloging. |
| CALL | Local dump extraction | low | Verify whether only cold/assert paths use it. |
| F2F | Local dump extraction | medium | Add conversion family when numeric conversion chapter is updated. |
| FADD | Local dump extraction | medium | Add scalar FP arithmetic row. |
| FFMA | Local dump extraction | medium | Add scalar FP fused multiply-add row. |
| FSEL | Local dump extraction | medium | Add scalar select row. |
| FSETP | Local dump extraction | medium | Add FP predicate compare row. |
| HFMA2 | Local dump extraction | medium | Add packed half arithmetic row. |
| I2FP | Local dump extraction | medium | Add integer-to-float conversion row. |
| IADD | Local dump extraction | medium | Add integer arithmetic row. |
| IMAD | Local dump extraction | medium | Add integer multiply-add row. |
| ISETP | Local dump extraction | medium | Add integer predicate compare row. |
| LDC | Local dump extraction | medium | Add constant load row and distinguish from `LDCU`. |
| LDL | Local dump extraction | medium | Add local-memory load row when spill chapter is refreshed. |
| LEA | Local dump extraction | medium | Add per-lane address arithmetic row. |
| LOP3 | Local dump extraction | medium | Add logical ternary operation row. |
| MOV | Local dump extraction | medium | Add register move row. |
| NOP | Local dump extraction | low | Add scheduling/padding row if useful for control-code work. |
| PLOP3 | Local dump extraction | low | Verify predicate logic context before cataloging. |
| PRMT | Local dump extraction | medium | Add byte/word permutation row. |
| R2P | Local dump extraction | low | Verify predicate-register movement context. |
| RET | Local dump extraction | low | Verify function-return context. |
| S2R | Local dump extraction | high | Add special-register read row. |
| SEL | Local dump extraction | medium | Add scalar select row. |
| SHF | Local dump extraction | medium | Add funnel-shift row. |
| STL | Local dump extraction | medium | Add local-memory store row when spill chapter is refreshed. |
| UFSEL | Local dump extraction | medium | Add uniform FP select row. |
| UI2FP | Local dump extraction | medium | Add uniform integer-to-float conversion row. |
| UIADD3 | Local dump extraction | medium | Add uniform integer add row. |
| UISETP | Local dump extraction | medium | Add uniform predicate compare row. |
| ULOP3 | Local dump extraction | medium | Add uniform logical ternary row. |
| UPLOP3 | Local dump extraction | low | Verify uniform predicate logic context. |
| USHF | Local dump extraction | medium | Add uniform shift row. |
| VOTEU | Local dump extraction | low | Verify relationship to `VOTE`. |

## Blackwell Watchlist

These families are listed as Blackwell additions in the seed list. Status is local-project status, not architectural truth. [GAP]

| Family | Local status | Priority | Notes |
|---|---|---|---|
| ACQSHMINIT | watchlist | low | Not observed locally. |
| CREDUX | watchlist | high | Mentioned as possible cluster-scope family in findings, not tested locally. |
| CS2UR | observed_uncataloged | high | Observed locally, needs glossary row. |
| FADD2 | watchlist | medium | Not observed locally. |
| FFMA2 | watchlist | medium | Not observed locally. |
| FHADD | watchlist | medium | Not observed locally. |
| FHFMA | watchlist | medium | Not observed locally. |
| FMNMX3 | watchlist | medium | Not observed locally. |
| FMUL2 | watchlist | medium | Not observed locally. |
| LDCU | documented | medium | Observed and cataloged. |
| LDT | watchlist | medium | Not observed locally. |
| LDTM | watchlist | medium | Not observed locally. |
| OMMA | documented | high | Observed and cataloged. |
| QMMA | encoding_page | high | Observed, cataloged, and has encoding page. |
| STT | watchlist | medium | Not observed locally. |
| STTM | watchlist | medium | Not observed locally. |
| UF2F | watchlist | medium | Not observed locally. |
| UF2I | watchlist | medium | Not observed locally. |
| UF2IP | watchlist | medium | Not observed locally. |
| UFADD | watchlist | medium | Not observed locally. |
| UFFMA | watchlist | medium | Not observed locally. |
| UFMNMX | watchlist | medium | Not observed locally. |
| UFMUL | watchlist | medium | Not observed locally. |
| UFRND | watchlist | medium | Not observed locally. |
| UFSEL | observed_uncataloged | medium | Observed locally, needs glossary row. |
| UFSET | watchlist | medium | Not observed locally. |
| UFSETP | watchlist | medium | Not observed locally. |
| UGETNEXTWORKID | watchlist | low | Not observed locally. |
| UI2F | watchlist | medium | Not observed locally. |
| UI2FP | observed_uncataloged | medium | Observed locally, needs glossary row. |
| UI2I | watchlist | medium | Not observed locally. |
| UI2IP | watchlist | medium | Not observed locally. |
| UIABS | watchlist | medium | Not observed locally. |
| UIMNMX | watchlist | medium | Not observed locally. |
| UMEMSETS | watchlist | low | Not observed locally. |
| UREDGR | watchlist | high | Reduction/writeback watchlist item. |
| USTGR | watchlist | high | Store/writeback watchlist item. |
| UTCATOMSWS | watchlist | high | Atomic-related watchlist item. |
| UTCBAR | watchlist | medium | Not observed locally. |
| UTCCP | watchlist | medium | Not observed locally. |
| UTCHMMA | watchlist | medium | Not observed locally. |
| UTCIMMA | watchlist | medium | Not observed locally. |
| UTCOMMA | watchlist | high | Related to OMMA path, not observed locally. |
| UTCQMMA | watchlist | high | Related to QMMA path, not observed locally. |
| UTCSHIFT | watchlist | medium | Not observed locally. |
| UVIADD | watchlist | medium | Not observed locally. |
| UVIMNMX | watchlist | medium | Not observed locally. |
| UVIRTCOUNT | watchlist | low | Not observed locally. |

## Removed Before Blackwell According To Seed

These are tracked so they are not accidentally treated as missing SM120 work. [INF]

| Generation | Removed families |
|---|---|
| Volta | `BFE`, `BFI`, `BRK`, `CAL`, `CONT`, `CSET`, `CSETP`, `DMNMX`, `DSET`, `FCMP`, `ICMP`, `IMADSP`, `ISET`, `JCAL`, `PBK`, `PCNT`, `PEXIT`, `PRET`, `PSET`, `RRO`, `SSY`, `SYNC`, `TEXS`, `TLD4S`, `TLDS`, `XMAD` |
| Ampere / Ada | `R2B`, `RTT` |
| Hopper | `RED` |
| Blackwell | `BGMMA`, `BMMA`, `HGMMA`, `IGMMA`, `QGMMA`, `ULDC`, `WARPGROUP`, `WARPGROUPSET` |

## Historical Seed Lists

The following seed lists are copied into the tracker as a watchlist source. They are not local observations. [GAP]

| Architecture band | Added or baseline families |
|---|---|
| Maxwell / Pascal baseline | `ATOM`, `ATOMS`, `B2R`, `BAR`, `BFE`, `BFI`, `BPT`, `BRA`, `BRK`, `BRX`, `CAL`, `CCTL`, `CCTLL`, `CCTLT`, `CONT`, `CS2R`, `CSET`, `CSETP`, `DADD`, `DFMA`, `DMNMX`, `DMUL`, `DSET`, `DSETP`, `EXIT`, `F2F`, `F2I`, `FADD`, `FCHK`, `FCMP`, `FFMA`, `FLO`, `FMNMX`, `FMUL`, `FSET`, `FSETP`, `FSWZADD`, `HADD2`, `HFMA2`, `HMUL2`, `HSET2`, `HSETP2`, `I2F`, `I2I`, `IADD`, `IADD3`, `ICMP`, `IMAD`, `IMADSP`, `IMNMX`, `IMUL`, `ISCADD`, `ISET`, `ISETP`, `JCAL`, `JMP`, `JMX`, `LD`, `LDC`, `LDG`, `LDL`, `LDS`, `LEA`, `LOP`, `LOP3`, `MEMBAR`, `MOV`, `MUFU`, `NOP`, `P2R`, `PBK`, `PCNT`, `PEXIT`, `POPC`, `PRET`, `PRMT`, `PSET`, `PSETP`, `R2B`, `R2P`, `RED`, `RET`, `RRO`, `S2R`, `SEL`, `SHF`, `SHFL`, `SHL`, `SHR`, `SSY`, `ST`, `STG`, `STL`, `STS`, `SUATOM`, `SULD`, `SURED`, `SUST`, `SYNC`, `TEX`, `TEXS`, `TLD`, `TLD4`, `TLD4S`, `TLDS`, `TXQ`, `VOTE`, `XMAD` |
| Volta additions | `ATOMG`, `BMOV`, `BMSK`, `BREAK`, `BREV`, `BSSY`, `BSYNC`, `CALL`, `DEPBAR`, `ERRBAR`, `FADD32I`, `FFMA32I`, `FMUL32I`, `FRND`, `FSEL`, `GETLMEMBASE`, `HADD2_32I`, `HFMA2_32I`, `HMMA`, `HMUL2_32I`, `I2IP`, `IABS`, `IADD32I`, `IDP`, `IDP4A`, `IMMA`, `IMUL32I`, `ISCADD32I`, `KILL`, `LEPC`, `LOP32I`, `MATCH`, `MOV32I`, `NANOSLEEP`, `PLOP3`, `PMTRIG`, `QSPC`, `RPCMOV`, `RTT`, `SETCTAID`, `SETLMEMBASE`, `SGXT`, `TMML`, `TXD`, `VABSDIFF`, `VABSDIFF4`, `WARPSYNC`, `YIELD` |
| Turing additions | `BMMA`, `BRXU`, `JMXU`, `LDSM`, `MOVM`, `R2UR`, `S2UR`, `UBMSK`, `UBREV`, `UCLEA`, `UFLO`, `UIADD3`, `UIADD3.64`, `UIMAD`, `UISETP`, `ULDC`, `ULEA`, `ULOP`, `ULOP3`, `ULOP32I`, `UMOV`, `UP2UR`, `UPLOP3`, `UPOPC`, `UPRMT`, `UPSETP`, `UR2UP`, `USEL`, `USGXT`, `USHF`, `USHL`, `USHR`, `VOTEU` |
| Ampere / Ada additions | `DMMA`, `F2IP`, `HMNMX2`, `I2FP`, `LDGDEPBAR`, `LDGSTS`, `REDUX`, `UF2FP` |
| Hopper additions | `ACQBULK`, `BGMMA`, `CGAERRBAR`, `ELECT`, `ENDCOLLECTIVE`, `FENCE`, `HGMMA`, `IGMMA`, `LDGMC`, `PREEXIT`, `QGMMA`, `REDAS`, `REDG`, `STAS`, `STSM`, `SYNCS`, `UBLKCP`, `UBLKPF`, `UBLKRED`, `UCGABAR_ARV`, `UCGABAR_WAIT`, `ULEPC`, `USETMAXREG`, `UTMACCTL`, `UTMACMDFLUSH`, `UTMALDG`, `UTMAPF`, `UTMAREDG`, `UTMASTG`, `VHMNMX`, `VIADD`, `VIADDMNMX`, `VIMNMX`, `VIMNMX3`, `WARPGROUP`, `WARPGROUPSET` |
| Blackwell additions | `ACQSHMINIT`, `CREDUX`, `CS2UR`, `FADD2`, `FFMA2`, `FHADD`, `FHFMA`, `FMNMX3`, `FMUL2`, `LDCU`, `LDT`, `LDTM`, `OMMA`, `QMMA`, `STT`, `STTM`, `UF2F`, `UF2I`, `UF2IP`, `UFADD`, `UFFMA`, `UFMNMX`, `UFMUL`, `UFRND`, `UFSEL`, `UFSET`, `UFSETP`, `UGETNEXTWORKID`, `UI2F`, `UI2FP`, `UI2I`, `UI2IP`, `UIABS`, `UIMNMX`, `UMEMSETS`, `UREDGR`, `USTGR`, `UTCATOMSWS`, `UTCBAR`, `UTCCP`, `UTCHMMA`, `UTCIMMA`, `UTCOMMA`, `UTCQMMA`, `UTCSHIFT`, `UVIADD`, `UVIMNMX`, `UVIRTCOUNT` |
| Reverse-engineered extras | `AL2P`, `ALD`, `ARRIVES`, `CSMTEST`, `F2FP`, `FOOTPRINT`, `IPA`, `ISBERD`, `LDTRAM`, `OUT`, `PIXLD`, `SUQUERY`, `USETSHMSZ` |

## Coverage Roadmap

1. Promote high-priority observed but uncataloged families into `SASS_INSTRUCTIONS_SM120.md`: `CS2UR`, `S2R`, `CS2R`, scalar arithmetic, local-memory spill loads/stores. [GAP]
2. Add a reduction/writeback page once enough `REDG`, `UREDGR`, `USTGR`, or `UTCATOMSWS` evidence exists. [GAP]
3. Add controlled probes only when a watchlist family is needed for production audit interpretation or fills a known roadmap gap. [INF]
4. Keep `SASS_INSTRUCTIONS_SM120.md` as the glossary for observed families, and keep this file as the completeness tracker. [INF]
