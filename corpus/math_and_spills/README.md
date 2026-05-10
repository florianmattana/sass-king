# Math And Spills

Chapters 11-12 cover compiler behavior that often dominates production-kernel audit work outside tensor-core hot paths.

| Chapter | Topic | Entry point |
|---|---|---|
| 11 | Division, transcendental slow paths, MUFU usage, and helper calls | [11_division_slow_path/conclusion_11.md](11_division_slow_path/conclusion_11.md) |
| 12 | Register pressure, local memory, spills, and frame allocation | [12_register_spill/conclusion_12.md](12_register_spill/conclusion_12.md) |

Read these when auditing kernels with unexpected instruction count growth, local memory traffic, or math-library expansion.
