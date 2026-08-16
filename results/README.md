# Published experiment results

Each terminal run is published under `results/runs/<run_id>/` by the controlled result
publisher. A run directory contains the final reports, aggregate statistics, a compact
state summary, and SHA-256 evidence manifest.

Raw per-query events, full logs, models, indexes, attack artifacts, and credentials are
never included by the automatic publisher.
