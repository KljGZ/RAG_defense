# RGRD-V0.1 Oracle mechanism-audit report

## Decision

**Gate V0.1: FAIL_TESTED**

The preregistered Oracle mechanism claim was tested and did not pass every corrected criterion.

This is a mechanism audit, not a deployable detector result. It does not establish a
runtime threshold, false-positive rate, or generalization beyond the frozen NQ
PoisonedRAG-B/W scope.

## Correction to V0

Legacy Gate 3 was not estimable because the preregistered family quotas were not
reached. H1--H3 were not executed. The confirmatory mechanism claim was therefore
inconclusive; it was not a tested failure. Legacy `T_RG` remains exploratory because
positive clipping and separate normalization discard absolute generation scale.

## Attrition ledger

| Family | Loaded samples | Unique queries | Canonical | Valid ranges | A/P co-chunk | Valid gold | Target distinct | Natural format | Forced format | Token partitions | 8 donor pairs | Finite mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PoisonedRAG-B | 479 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 98 | 98 | 98 |
| PoisonedRAG-W | 497 | 100 | 100 | 100 | 100 | 100 | 100 | 95 | 92 | 92 | 92 | 92 |

Canonicalization selected the lexicographically smallest `sample_id` per `query_id`
before observing retrieval, natural generation, forced generation, or any mechanism
score. Retrieval/generation outcomes were retained as labels and never as filters.

## Primary family-specific tests

| Test | n | Positive proportion | Median | Query-bootstrap 95% CI | Exact sign p | Holm p | Pass |
|---|---:|---:|---:|---|---:|---:|---|
| PoisonedRAG-B/C_R | 98 | 0.5612 | 0.1091 | [-0.07694, 0.6052] | 0.1332 | 0.2664 | False |
| PoisonedRAG-B/C_G | 98 | 0.9898 | 5.626 | [3.655, 7.549] | 3.124e-28 | 1.25e-27 | True |
| PoisonedRAG-W/C_R | 92 | 0.1087 | -4.547 | [-5.909, -3.479] | 1 | 1 | False |
| PoisonedRAG-W/C_G | 92 | 0.9022 | 4.554 | [2.773, 6.859] | 1.962e-16 | 5.885e-16 | True |

Each test requires both Holm-adjusted p < 0.05 and a strictly positive lower
query-bootstrap confidence bound. `C_R = phi_A^R - phi_P^R`; `C_G = phi_P^G - phi_A^G`.
No clipping, per-channel normalization, cross-channel comparison, or combined score
was used.

## Attention-hole direction robustness

**Robustness gate: FAIL_TESTED**

| Family | Mask-valid n | Retrieval agreement | Generation agreement | Pass |
|---|---:|---:|---:|---|
| PoisonedRAG-B | 98 | 0.7385 | 0.9337 | False |
| PoisonedRAG-W | 92 | 0.9552 | 0.928 | True |

The attention-hole analysis is direction-only. The eight same-token-length benign
donor pairs remain the primary estimator.

## Outcome association (secondary)

| Contrast / outcome | Status | Outcome=1 | Outcome=0 | Median difference | Cliff's delta | Permutation p |
|---|---|---:|---:|---:|---:|---:|
| PoisonedRAG-B/C_R~R | NOT_ESTIMABLE | NA | NA | NA | NA | NA |
| PoisonedRAG-B/C_G~G_f | ESTIMATED_SECONDARY | 42 | 56 | 6.372 | 0.5102 | 0.0002 |
| PoisonedRAG-W/C_R~R | ESTIMATED_SECONDARY | 91 | 1 | -0.7837 | -0.1648 | 0.5911 |
| PoisonedRAG-W/C_G~G_f | ESTIMATED_SECONDARY | 34 | 58 | 6.905 | 0.5477 | 0.0003 |

These family-specific associations test whether `C_R` tracks actual reranker survival
and whether `C_G` tracks fixed-position forced-context attack success. They do not
replace the primary mechanism tests.

## Frozen contracts and claim boundary

- Unit: one final chunk per independent query.
- Context: clean Top-K with the poison fixed at zero-based index 1 for generation attribution.
- Models: frozen Contriever, MiniLM cross-encoder, and Qwen2.5-7B-Instruct.
- Token ownership: maximum character overlap, with boundary ties assigned in
  deterministic A-then-P order; one model token can belong to at most one player.
- Generator: explicit BF16, eager attention, strict deterministic algorithms, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- GPU boundary: physical GPUs 4--7 only; physical GPUs 0--3 excluded.
- Excluded claims: Phantom, Joint-GCG, multi-chunk attacks, latent backdoors,
  GraphRAG, multimodal RAG, and other attack families.

Generated at `2026-08-16T16:50:16.499281+00:00` from commit
`10eab757e6387b4f8e69a477a9ee67b8681a780d`.
