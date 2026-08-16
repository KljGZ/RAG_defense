# RGRD-V0.1 frozen mechanism-audit protocol

V0.1 corrects the interpretation and estimator used by V0.  The legacy V0 run
did not execute H1--H3 because its preregistered family quotas were not met;
therefore its Gate 3 status is `NOT_ESTIMABLE`, not a tested mechanism failure.
Legacy artifacts are preserved and never overwritten.

V0.1 is an Oracle mechanism audit, not a deployable detector.  It uses one
outcome-independent canonical attack sample per query and retains every query
that passes only format, answer-validity, and anchor/payload co-chunk checks.
Retrieval and generation outcomes are measured after sampling and are never
used to select observations.

For the two Oracle players anchor (A) and payload (P), V0.1 evaluates the four
coalitions empty, A, P, and A+P.  The retrieval value is the frozen two-stage
bottleneck margin.  The generation value is target mean token log-probability
minus the mean token log-probability of a gold alias selected once in the clean
context and then fixed for all coalitions.  Two-player Shapley values define
the scale-preserving contrasts

`C_R = phi_A^R - phi_P^R` and `C_G = phi_P^G - phi_A^G`.

The primary intervention is the median of eight deterministic, exact-token-
length, benign donor-pair replacements.  At least six pairs must be finite.
Attention-hole masking is a direction-only robustness check.  There is no
positive clipping, per-channel normalization, combined score, or reuse of the
legacy `T_RG` statistic.

The independent unit is the query.  Four family-specific one-sided exact sign
tests (B/W crossed with `C_R`/`C_G`) receive Holm correction.  Each claim also
requires a query-bootstrap median confidence interval with lower endpoint
above zero.  Fewer than 30 identifiable queries in a family yields
`NOT_ESTIMABLE` rather than `FAIL_TESTED`.

All V0.1 tasks are restricted to physical GPUs 4--7.  Physical GPUs 0--3 are a
hard exclusion and the runner refuses to start if the configuration differs.
