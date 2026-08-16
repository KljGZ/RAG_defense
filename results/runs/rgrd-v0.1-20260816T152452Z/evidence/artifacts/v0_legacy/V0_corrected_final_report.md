# Corrected interpretation of the frozen RGRD-V0 run

Generated at `2026-08-16T15:24:54.013216+00:00`. The original `artifacts/reports/V0_final_report.md` is
preserved byte-for-byte and is not replaced by this correction.

## Corrected Gate 3 status: NOT_ESTIMABLE

Gate 3 was not estimable because the preregistered family quotas were not reached.
H1--H3 were not executed. The confirmatory mechanism claim is therefore inconclusive.

- Observed successful-event counts: `{"PoisonedRAG-B": 92, "PoisonedRAG-W": 89}`
- Recorded quota shortfalls: `{"PoisonedRAG-B": {"available": 92, "required": 100}, "PoisonedRAG-W": {"available": 89, "required": 100}}`

This is not evidence that the mechanism hypothesis was tested and failed. It is an
estimability failure caused by the V0 success-conditioned quota design.

## Estimator limitation

The legacy `T_RG` statistic applies positive clipping and separately normalizes the
retrieval and generation maps. It therefore discards absolute effect scale: a chunk
with negligible generation mass can still obtain an extreme `T_RG`. The observed
legacy role direction was mixed, and no confirmatory conclusion may be drawn from it.

The frozen evidence remains useful only for explicitly exploratory diagnostics. V0.1
uses scale-preserving Oracle A/P Shapley contrasts and an outcome-independent query
sample instead.
