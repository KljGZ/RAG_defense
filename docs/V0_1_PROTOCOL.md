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

The primary intervention is the median of eight deterministic, benign donor-pair
replacements. Fast-tokenizer boundary pieces are assigned to exactly one Oracle
span: maximum character overlap wins, and ties go to the earlier ordered player
(all A spans before all P spans). This ownership partition is recomputed under each
frozen model tokenizer and recorded per query, preventing a boundary subword from
being counted as both players.

Donor length is exact under the frozen generator tokenizer. Under Contriever and
the cross-encoder, the donor must contain at least the corresponding partition
length and its deterministic token-ID prefix is used. Replacement is applied to
fixed token positions rather than by reconstructing the chunk text, so every
coalition preserves each model's sequence length, attention length, and positional
indices. A span with no uniquely owned token, or a query for which eight valid donor
pairs cannot be drawn, is written to the attrition ledger; it is never silently
coerced or selected using retrieval/generation outcomes. At least six of the eight
executed pairs must yield finite contrasts.

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
