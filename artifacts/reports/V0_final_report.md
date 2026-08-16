# RGRD-V0 Final Report

Generated: `2026-08-16T09:47:28.699498+00:00`
Run status: **FAILED**.

RGRD is evaluated here as a structural anomaly screener at the post-reranking/pre-release decision point, not as a universal malicious-content classifier.

## Post-hoc scope warning

This run uses `PA-001-exclude-phantom` after `rgrd-v0-20260814T190433Z` failed Gate 1. Active attack families: PoisonedRAG-B, PoisonedRAG-W; excluded: Phantom.
Claim boundary: Post-hoc PoisonedRAG-B/W-only evidence; no conclusion may be generalized to Phantom or all modular RAG attacks.

## Eight required decisions

1. **Are the in-scope modular attack reproductions credible?** PoisonedRAG=PASS_FUNCTIONAL; Gate 1=PASS. Phantom was audited as WARN but excluded under PA-001-exclude-phantom.
2. **Is the oracle anchor retrieval-dominant?** Not estimable: Gate-3 H1 was not completed.
3. **Is the oracle payload generation-dominant?** Not estimable: Gate-3 H2 was not completed.
4. **Is poison T_RG higher than matched clean?** Not estimable: poison-versus-matched-clean H3 was not completed.
5. **What is detection at 5% query-FPR?** Not estimable: conformal detection was not reached or did not finish.
6. **Which benign structures cause false positives?** Not estimable: hard-benign evaluation was not reached or did not finish.
7. **Does Joint-GCG weaken role separation?** Not estimable: the Joint-GCG boundary phase was not reached or did not finish.
8. **Continue, narrow, or terminate?** Terminate the confirmatory detector path: the preregistered role-decoupling mechanism was not established.

## Gate discipline

No result from a phase after a failed gate is promoted as confirmatory. Missing values are reported as not estimable rather than imputed. This post-hoc scope amendment is never represented as part of the original preregistration.

## Terminal reason

GateFailure: gate_3 failed: successful co-chunk/reranked/end-to-end event quota not met; Traceback (most recent call last):
  File "/mnt/data/jkl/RGRD_V0/rgrd/experiments/run.py", line 1505, in run
    progress = function()
  File "/mnt/data/jkl/RGRD_V0/rgrd/experiments/run.py", line 1112, in _phase_5
    raise GateFailure("gate_3", reasons)
GateFailure: gate_3 failed: successful co-chunk/reranked/end-to-end event quota not met

