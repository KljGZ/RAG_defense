# RGRD-V0 Pipeline Report

Generated: `2026-08-16T09:47:28.659576+00:00`
Run ID: `rgrd-v0-20260816T080158Z`; status: **FAILED**.

## Fixed Track-B pipeline

- Generator-token chunks: 256 tokens, overlap 32.
- Retrieval: exact `IndexFlatIP`, dense Top-50, reranked Top-5.
- Interventions: 32-token spans at offsets [0, 16].
- Generation: temperature=0.0, do_sample=False, max_new_tokens=32.
- Decision point: after reranking and deterministic shadow generation, before answer release.
- Runtime detector view excludes attack labels, target/gold answers, and all oracle ranges.

## Post-hoc protocol scope amendment

- Protocol: `rgrd-v0-poisonedrag-only-amendment-1`; amendment: `PA-001-exclude-phantom`.
- Prior failed run retained as evidence: `rgrd-v0-20260814T190433Z`.
- Active modular attack families: PoisonedRAG-B, PoisonedRAG-W.
- Excluded modular attack families: Phantom.
- Claim boundary: Post-hoc PoisonedRAG-B/W-only evidence; no conclusion may be generalized to Phantom or all modular RAG attacks.

## Pinned models

- retriever: `facebook/contriever` at revision `2bd46a25019aeea091fd42d1f0fd4801675cf699`.
- reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- generator: `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`.

## Provenance locks

- Model lock: `/mnt/data/jkl/RGRD_V0/configs/models.lock.yaml` (SHA-256 `662f62667ab1e0f9fa12ee6a4e0483cb7b4253c16c6bd2fd63014bc4ef3ae67b`).
- Upstream lock: `/mnt/data/jkl/RGRD_V0/third_party/manifest.lock.yaml` (SHA-256 `414100a061bea418eff9210bade90070682073f4105c5713c2163ad44694d988`).
- `/home/jkl/RAG_attack` was used as read-only attack evidence.
- Each upstream paper project has a separate Conda environment declaration.

## Gate decisions

- gate_1: PASS
  - poisonedrag status is PASS_FUNCTIONAL
  - phantom audited as WARN but is informational-only under PA-001-exclude-phantom
- gate_2: PASS
  - all 20 queries passed every exact check
- gate_3: FAIL
  - successful co-chunk/reranked/end-to-end event quota not met
- gate_robustness: not decided

## Phase ledger

| Phase | Status | Progress / error |
|---|---|---|
| phase_0_provenance | passed | `{"active_command": "phase0-pytest", "active_group": "phase0-model-preflight", "active_pid": 2077298, "environment": "/mnt/data/jkl/RGRD_V0/artifacts/provenance/rgrd_environment.json", "gpu_admission": {"allowed_physical_gpus": [4, 5, 6, 7], "base_min_free_mib": 19000, "completed": ["single"], "excluded_physical_gpus": [0, 1, 2, 3], "gpu_memory_mib": {"0": {"free": 10742, "total": 24564}, "1": {"free": 8022, "total": 24564}, "2": {"free": 8704, "total": 24564}, "3": {"free": 8258, "total": 24564}, "4": {"free": 19456, "total": 24564}, "5": {"free": 19504, "total": 24564}, "6": {"free": 19584, "total": 24564}, "7": {"free": 19656, "total": 24564}}, "oom_retries": {"single": 0}, "pending": [], "quarantined_gpus": [], "required_free_mib": {"single": 19000}, "running": {}}, "heartbeat": "2026-08-16T08:47:33.460044+00:00", "joint_environment": "/mnt/data/jkl/RGRD_V0/artifacts/provenance/joint_gcg_environment.json", "model_runtime_lock": "/mnt/data/jkl/RGRD_V0/artifacts/provenance/models.runtime.yaml", "models": "passed", "protocol_scope": {"active_attack_families": ["PoisonedRAG-B", "PoisonedRAG-W"], "amendment_id": "PA-001-exclude-phantom", "amendment_reason": "User-directed scope narrowing after Gate 1 found no non-empty optimized Phantom s_gen span.", "claim_boundary": "Post-hoc PoisonedRAG-B/W-only evidence; no conclusion may be generalized to Phantom or all modular RAG attacks.", "detection_quotas": {"PoisonedRAG-B": 50, "PoisonedRAG-W": 50}, "excluded_attack_families": ["Phantom"], "mechanism_quotas": {"PoisonedRAG-B": 100, "PoisonedRAG-W": 100}, "prior_failed_run": "rgrd-v0-20260814T190433Z", "protocol_id": "rgrd-v0-poisonedrag-only-amendment-1"}, "tests": "passed", "upstream_runtime_lock": "/mnt/data/jkl/RGRD_V0/artifacts/provenance/upstreams.runtime.yaml"}` |
| phase_1_attack_audit | passed | `{"active_command": "phase1-reproduction-audit", "active_group": "phase1-poisonedrag-whitebox-smoke", "active_pid": 2082004, "audit": "PASS", "gpu_admission": {"allowed_physical_gpus": [4, 5, 6, 7], "base_min_free_mib": 19000, "completed": ["single"], "excluded_physical_gpus": [0, 1, 2, 3], "gpu_memory_mib": {"0": {"free": 10742, "total": 24564}, "1": {"free": 8022, "total": 24564}, "2": {"free": 8704, "total": 24564}, "3": {"free": 8258, "total": 24564}, "4": {"free": 19456, "total": 24564}, "5": {"free": 19504, "total": 24564}, "6": {"free": 19584, "total": 24564}, "7": {"free": 19656, "total": 24564}}, "oom_retries": {"single": 0}, "pending": [], "quarantined_gpus": [], "required_free_mib": {"single": 19000}, "running": {}}, "heartbeat": "2026-08-16T08:50:04.481570+00:00", "whitebox_smoke": 20}` |
| phase_2_deterministic_pipeline | passed | `{"active_command": "phase2-pytest", "active_group": "phase2-determinism", "active_pid": 2082394, "determinism_queries": 20, "gpu_admission": {"allowed_physical_gpus": [4, 5, 6, 7], "base_min_free_mib": 19000, "completed": ["single"], "excluded_physical_gpus": [0, 1, 2, 3], "gpu_memory_mib": {"0": {"free": 10742, "total": 24564}, "1": {"free": 8022, "total": 24564}, "2": {"free": 8704, "total": 24564}, "3": {"free": 8258, "total": 24564}, "4": {"free": 19456, "total": 24564}, "5": {"free": 19504, "total": 24564}, "6": {"free": 19584, "total": 24564}, "7": {"free": 19656, "total": 24564}}, "oom_retries": {"single": 0}, "pending": [], "quarantined_gpus": [], "required_free_mib": {"single": 19000}, "running": {}}, "heartbeat": "2026-08-16T08:56:06.936757+00:00", "nq_index_chunks": 2844238}` |
| phase_3_span_intervention | passed | `{"active_group": "phase3-noop-noise", "generation_floor": 1e-08, "gpu_admission": {"allowed_physical_gpus": [4, 5, 6, 7], "base_min_free_mib": 19000, "completed": ["single"], "excluded_physical_gpus": [0, 1, 2, 3], "gpu_memory_mib": {"0": {"free": 10742, "total": 24564}, "1": {"free": 8022, "total": 24564}, "2": {"free": 8704, "total": 24564}, "3": {"free": 8258, "total": 24564}, "4": {"free": 19456, "total": 24564}, "5": {"free": 19504, "total": 24564}, "6": {"free": 19584, "total": 24564}, "7": {"free": 19656, "total": 24564}}, "oom_retries": {"single": 0}, "pending": [], "quarantined_gpus": [], "required_free_mib": {"single": 19000}, "running": {}}, "heartbeat": "2026-08-16T09:02:09.529771+00:00", "observations": 100, "retrieval_floor": 4.1477268840438114e-05}` |
| phase_4_role_attribution | passed | `{"active_group": "phase4-attribution-pilot", "gpu_admission": {"allowed_physical_gpus": [4, 5, 6, 7], "base_min_free_mib": 19000, "completed": ["single"], "excluded_physical_gpus": [0, 1, 2, 3], "gpu_memory_mib": {"0": {"free": 10742, "total": 24564}, "1": {"free": 8022, "total": 24564}, "2": {"free": 8704, "total": 24564}, "3": {"free": 8258, "total": 24564}, "4": {"free": 19456, "total": 24564}, "5": {"free": 19504, "total": 24564}, "6": {"free": 19584, "total": 24564}, "7": {"free": 19656, "total": 24564}}, "oom_retries": {"single": 0}, "pending": [], "quarantined_gpus": [], "required_free_mib": {"single": 19000}, "running": {}}, "heartbeat": "2026-08-16T09:04:40.887811+00:00", "pilot_queries": 5, "runtime_view": "label-free"}` |
| phase_5_oracle_mechanism | failed | `{"active_command": "phase5-combine-mechanism", "active_group": "mechanism-poisonedrag_w", "active_pid": 2093414, "gpu_admission": {"allowed_physical_gpus": [4, 5, 6, 7], "base_min_free_mib": 19000, "completed": ["shard-00", "shard-01", "shard-02", "shard-03", "shard-04", "shard-05", "shard-06", "shard-07"], "excluded_physical_gpus": [0, 1, 2, 3], "gpu_memory_mib": {"0": {"free": 12308, "total": 24564}, "1": {"free": 10608, "total": 24564}, "2": {"free": 8704, "total": 24564}, "3": {"free": 8258, "total": 24564}, "4": {"free": 19456, "total": 24564}, "5": {"free": 19504, "total": 24564}, "6": {"free": 19584, "total": 24564}, "7": {"free": 19656, "total": 24564}}, "oom_retries": {"shard-00": 0, "shard-01": 0, "shard-02": 0, "shard-03": 0, "shard-04": 0, "shard-05": 0, "shard-06": 0, "shard-07": 0}, "pending": [], "quarantined_gpus": [], "required_free_mib": {"shard-00": 19000, "shard-01": 19000, "shard-02": 19000, "shard-03": 19000, "shard-04": 19000, "shard-05": 19000, "shard-06": 19000, "shard-07": 19000}, "running": {}}, "heartbeat": "2026-08-16T09:46:58.561675+00:00"}; error=GateFailure: gate_3 failed: successful co-chunk/reranked/end-to-end event quota not met` |
| phase_6_intervention_robustness | skipped | `{}; error=not run because a preceding gate or phase failed` |
| phase_7_conformal_detection | skipped | `{}; error=not run because a preceding gate or phase failed` |
| phase_8_hard_benign | skipped | `{}; error=not run because a preceding gate or phase failed` |
| phase_9_joint_gcg_boundary | skipped | `{}; error=not run because a preceding gate or phase failed` |
