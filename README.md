# RGRD-V0

RGRD-V0 validates whether successful modular RAG poisoning leaves a recoverable
within-chunk separation between retrieval-dominant and generation-dominant spans.
It is a **structural anomaly screener**, not a final malicious-content classifier.

## Threat model and decision point

- Unit inspected: the final chunk that survives dense retrieval and reranking.
- Internal attribution unit: deterministic equal-token spans (32 tokens, views 0 and 16).
- Decision point: post-reranking, after deterministic shadow generation, before release.
- Defender: white-box retriever, reranker, and open generator.
- Runtime detector inputs exclude attack labels, target/gold answers, and oracle spans.
- Oracle anchor/payload boundaries are used only for mechanism validation.

The primary statistic is

`T_RG = 0.5 * sum(abs(pR - pG))`,

where `pR` and `pG` are normalized positive retrieval and generation causal effects.
If either effect mass is below a no-op noise floor, the role is unidentifiable and
`T_RG` is conservatively set to zero.

## Gate discipline

1. **Gate 1:** PoisonedRAG is at least functionally reproduced and Phantom exposes
   independent retrieval/generation/command/trigger components.
2. **Gate 2:** a 20-query end-to-end run is deterministic, including Top-K, shadow
   answers, and source-to-chunk-to-token offsets.
3. **Gate 3:** H1-H3 pass one-sided paired sign-flip tests after Holm correction and
   all query-cluster bootstrap confidence intervals lie above zero.

No later experiment may be promoted to a confirmatory result when its preceding gate
fails. Failure reports remain first-class deliverables.

## Safety and provenance

- `/home/jkl/RAG_attack` is read-only evidence. RGRD never resets or edits it.
- Official projects are complete independent clones under `third_party/` and are pinned
  by commit in `third_party/manifest.lock.yaml`.
- Each paper project has an independent environment declaration under `envs/`.
- Attacks run only on local public benchmarks.

## Core commands

```bash
python -m rgrd.audit.scan \
  --root /home/jkl/RAG_attack \
  --upstream-manifest configs/upstreams.yaml \
  --output artifacts/audit

pytest -q
```

## Reproducible environments

- Main pipeline: `/mnt/data/jkl/conda-envs/rgrd-v0`
- Joint-GCG boundary: `/mnt/data/jkl/conda-envs/rgrd-v0-joint-gcg`
- Eight upstream paper environments are declared separately in `envs/`; none is
  merged into the RGRD environment.

The generator is the pinned `Qwen/Qwen2.5-7B-Instruct` revision recorded in
`configs/models.lock.yaml`. Model files, complete upstream clones, indexes, attacks,
and raw events are runtime data and are intentionally excluded from Git.

## Gate-controlled full run

The runner is restart-safe only within the same committed detector/model/config
provenance. Worker rows from another commit are rejected instead of silently reused.

```bash
cd /home/jkl/RGRD_V0
PYTHONDONTWRITEBYTECODE=1 \
  /mnt/data/jkl/conda-envs/rgrd-v0/bin/python -m rgrd.experiments.run \
  --root /home/jkl/RGRD_V0 \
  --attack-root /home/jkl/RAG_attack \
  --joint-python /mnt/data/jkl/conda-envs/rgrd-v0-joint-gcg/bin/python
```

The runner writes `artifacts/run_state.json` atomically. It stops at the first failed
gate, marks all later phases skipped, and still writes the required failure-aware
reports. Read-only status inspection is:

```bash
/mnt/data/jkl/conda-envs/rgrd-v0/bin/python -m rgrd.pipeline.monitor \
  --state /home/jkl/RGRD_V0/artifacts/run_state.json
```

The pinned Joint-GCG clone is never edited. Its upstream model loader omits parameter
freezing and its success branch returns before JSON persistence; the project-side
wrapper applies only those two non-algorithmic compatibility guards. Losses,
coordinate gradients, candidate sampling, optimizer schedule, and stopping rules
remain upstream behavior and the guards are recorded in every worker manifest.

## Required terminal reports

Successful and gate-stopped runs both materialize:

- `artifacts/audit/reproduction_audit.md`
- `artifacts/reports/pipeline_report.md`
- `artifacts/reports/mechanism_report.md`
- `artifacts/reports/detection_report.md`
- `artifacts/reports/hard_benign_report.md`
- `artifacts/reports/joint_gcg_boundary_report.md`
- `artifacts/reports/V0_final_report.md`
