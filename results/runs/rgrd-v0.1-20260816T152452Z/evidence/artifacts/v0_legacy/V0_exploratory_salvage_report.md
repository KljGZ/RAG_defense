# V0 exploratory salvage (`EXPLORATORY_ONLY`)

This document aggregates the frozen, success-conditioned V0 events by independent
query. It is not a gate, does not select a threshold, and cannot support a
confirmatory mechanism claim.

| Family | Query n | Spearman(T_RG, poison generation mass) | Spearman(T_RG, clean generation mass) | Clean low-mass and T_RG>0.8 |
|---|---:|---:|---:|---:|
| PoisonedRAG-B | 27 | 0.119 | 0.2186 | 0 |
| PoisonedRAG-W | 29 | 0.3217 | -0.0004926 | 0 |

"Low generation mass" is defined here, before counting, as no larger than the frozen
V0 deterministic no-op generation floor (`1e-08`). The diagnostic directly
checks the scale pathology: an extreme normalized total-variation score can coexist
with negligible absolute generation effect.

Rows read: 181; query-level rows after aggregation: 56.
