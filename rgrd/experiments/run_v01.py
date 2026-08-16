from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rgrd.experiments.gpu_admission import (
    log_segment_has_cuda_oom,
    query_gpu_memory,
    rank_eligible_gpus,
)
from rgrd.pipeline.state import (
    GateDecision,
    GateStatus,
    PhaseRecord,
    RunState,
    StateStore,
)
from rgrd.provenance import sha256_file, utc_now
from rgrd.publishing.github import publish_terminal_results


_ALLOWED = [4, 5, 6, 7]
_EXCLUDED = [0, 1, 2, 3]


@dataclass
class WorkerTask:
    family: str
    shard_id: int
    shards: int
    manifest: Path
    output: Path
    log: Path
    retries: int = 0
    not_before: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.family}/shard-{self.shard_id:02d}-of-{self.shards:02d}"


@dataclass
class RunningTask:
    task: WorkerTask
    physical_gpu: int
    process: subprocess.Popen[Any]
    log_handle: Any
    log_start_offset: int
    started_at: str = field(default_factory=utc_now)


class V01Runner:
    def __init__(
        self,
        *,
        root: Path,
        attack_root: Path,
        state_path: Path,
        shards: int,
    ) -> None:
        self.root = root.resolve()
        self.attack_root = attack_root.resolve()
        self.state_path = state_path.resolve()
        self.shards = shards
        self.config = yaml.safe_load(
            (self.root / "configs/pipeline/v0_1.yaml").read_text(encoding="utf-8")
        )
        self.protocol = yaml.safe_load(
            (self.root / "configs/experiments/v0_1_preregistration.yaml").read_text(
                encoding="utf-8"
            )
        )
        admission = self.config["runtime"]["gpu_admission"]
        if list(admission["allowed_physical_gpus"]) != _ALLOWED:
            raise RuntimeError("V0.1 runner requires allowed_physical_gpus=[4,5,6,7]")
        if list(admission["excluded_physical_gpus"]) != _EXCLUDED:
            raise RuntimeError("V0.1 runner requires excluded_physical_gpus=[0,1,2,3]")
        if self.shards < 1:
            raise ValueError("shards must be positive")
        self.allowed = list(_ALLOWED)
        self.excluded = list(_EXCLUDED)
        self.minimum_free = int(admission["full_model_min_free_mib"])
        self.poll_seconds = int(admission["poll_seconds"])
        self.retry_limit = int(admission["oom_retry_limit"])
        self.retry_cooldown = int(admission["oom_retry_cooldown_seconds"])
        self.retry_headroom = int(admission["oom_retry_headroom_mib"])
        self.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.models = {
            role: str(value["revision"]) for role, value in self.config["models"].items()
        }
        self.run_id = "rgrd-v0.1-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.store = StateStore(self.state_path)
        self.running: dict[int, RunningTask] = {}
        self.stop_requested = False
        self.log_path = self.root / "artifacts/logs/runner.log"

    def _log(self, message: str) -> None:
        print(f"[{utc_now()}] {message}", flush=True)

    def _archive_prior_state(self) -> None:
        if not self.state_path.is_file():
            return
        archive = self.root / "artifacts/v0_legacy/run_state_before_v01.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            shutil.copy2(self.state_path, archive)

    def _initial_state(self) -> RunState:
        phases = {
            name: PhaseRecord(phase=name)
            for name in (
                "freeze_v0_legacy",
                "prepare_v01_manifests",
                "oracle_shapley_workers",
                "combine_v01_statistics",
            )
        }
        return RunState(
            schema_version=3,
            run_id=self.run_id,
            status="running",
            root=str(self.root),
            detector_code_commit=self.commit,
            model_revisions=self.models,
            pipeline_config_sha256=sha256_file(
                self.root / "configs/pipeline/v0_1.yaml"
            ),
            experiment_config_sha256=sha256_file(
                self.root / "configs/experiments/v0_1_preregistration.yaml"
            ),
            pid=os.getpid(),
            log_path=str(self.log_path),
            phases=phases,
        )

    def _set_phase(self, name: str, status: str, **progress: Any) -> None:
        def mutate(state: RunState) -> None:
            record = state.phases[name]
            state.current_phase = name
            record.status = status
            if status == "running" and record.started_at is None:
                record.started_at = utc_now()
            if status in {"passed", "failed", "skipped"}:
                record.finished_at = utc_now()
            record.progress.update(progress)

        self.store.mutate(mutate)

    def _fail(self, message: str) -> None:
        def mutate(state: RunState) -> None:
            state.status = "failed"
            state.errors.append(message)
            if state.current_phase:
                record = state.phases[state.current_phase]
                record.status = "failed"
                record.error = message
                record.finished_at = utc_now()

        self.store.mutate(mutate)
        self._log(f"FAILED: {message}")

    def _publish_terminal(self) -> bool:
        try:
            result = publish_terminal_results(self.root, self.state_path)
        except Exception as exc:
            message = f"terminal result publication failed: {type(exc).__name__}: {exc}"

            def failed(state: RunState) -> None:
                state.publication = {
                    "status": "failed",
                    "error": message,
                    "at": utc_now(),
                }
                state.errors.append(message)

            self.store.mutate(failed)
            self._log(message)
            return False

        def succeeded(state: RunState) -> None:
            state.publication = dict(result)

        self.store.mutate(succeeded)
        self._log(
            f"publication status={result['status']} branch={result.get('branch')} "
            f"commit={result.get('commit')} destination={result.get('destination')}"
        )
        return result["status"] in {"published", "already_published"}

    def _run_cpu_command(self, label: str, command: list[str]) -> None:
        log = self.root / f"artifacts/logs/{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
        self._log(f"start {label}: {' '.join(command)}")
        with log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=self.root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{label} exited with code {completed.returncode}; log={log}")

    def freeze_legacy(self) -> None:
        name = "freeze_v0_legacy"
        self._set_phase(name, "running", heartbeat=utc_now())
        self._run_cpu_command(
            "v01-freeze-v0",
            [sys.executable, "-m", "rgrd.experiments.legacy_v0", "--root", str(self.root)],
        )
        self._set_phase(
            name,
            "passed",
            heartbeat=utc_now(),
            corrected_report=str(
                self.root / "artifacts/v0_legacy/V0_corrected_final_report.md"
            ),
            exploratory_report=str(
                self.root / "artifacts/v0_legacy/V0_exploratory_salvage_report.md"
            ),
        )

    def prepare_manifests(self) -> dict[str, Path]:
        name = "prepare_v01_manifests"
        self._set_phase(name, "running", heartbeat=utc_now())
        manifests: dict[str, Path] = {}
        for family in self.protocol["scope"]["active_attack_families"]:
            safe = family.lower().replace("-", "_")
            output = self.root / f"artifacts/v01/manifests/{safe}.json"
            self._run_cpu_command(
                f"v01-prepare-{safe}",
                [
                    sys.executable,
                    "-m",
                    "rgrd.experiments.v01_prepare",
                    "--root",
                    str(self.root),
                    "--attack-root",
                    str(self.attack_root),
                    "--family",
                    family,
                    "--dataset",
                    "nq",
                    "--output",
                    str(output),
                ],
            )
            manifests[family] = output
        self._set_phase(
            name,
            "passed",
            heartbeat=utc_now(),
            manifests={family: str(path) for family, path in manifests.items()},
        )
        return manifests

    def _tasks(self, manifests: dict[str, Path]) -> list[WorkerTask]:
        tasks: list[WorkerTask] = []
        families = list(self.protocol["scope"]["active_attack_families"])
        for shard_id in range(self.shards):
            for family in families:
                safe = family.lower().replace("-", "_")
                tasks.append(
                    WorkerTask(
                        family=family,
                        shard_id=shard_id,
                        shards=self.shards,
                        manifest=manifests[family],
                        output=self.root
                        / f"artifacts/v01/events/{safe}/shard-{shard_id:02d}.jsonl",
                        log=self.root
                        / f"artifacts/logs/v01-{safe}-shard-{shard_id:02d}.log",
                    )
                )
        return tasks

    def _launch(self, task: WorkerTask, physical_gpu: int) -> RunningTask:
        task.log.parent.mkdir(parents=True, exist_ok=True)
        task.output.parent.mkdir(parents=True, exist_ok=True)
        offset = task.log.stat().st_size if task.log.exists() else 0
        handle = task.log.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        command = [
            sys.executable,
            "-m",
            "rgrd.experiments.v01_worker",
            "--root",
            str(self.root),
            "--attack-root",
            str(self.attack_root),
            "--family",
            task.family,
            "--dataset",
            "nq",
            "--manifest",
            str(task.manifest),
            "--output",
            str(task.output),
            "--shard-id",
            str(task.shard_id),
            "--shards",
            str(task.shards),
            "--device",
            "cuda:0",
            "--detector-commit",
            self.commit,
        ]
        process = subprocess.Popen(
            command,
            cwd=self.root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._log(
            f"launched {task.key} pid={process.pid} physical_gpu={physical_gpu} "
            f"retry={task.retries}"
        )
        return RunningTask(
            task=task,
            physical_gpu=physical_gpu,
            process=process,
            log_handle=handle,
            log_start_offset=offset,
        )

    def _worker_progress(
        self,
        *,
        pending: list[WorkerTask],
        completed: list[str],
        memory: dict[int, Any],
        retries: dict[str, int],
    ) -> dict[str, Any]:
        return {
            "heartbeat": utc_now(),
            "allowed_physical_gpus": self.allowed,
            "excluded_physical_gpus": self.excluded,
            "minimum_free_mib": self.minimum_free,
            "pending": [task.key for task in pending],
            "running": {
                running.task.key: {
                    "pid": running.process.pid,
                    "physical_gpu": running.physical_gpu,
                    "started_at": running.started_at,
                    "retry": running.task.retries,
                }
                for running in self.running.values()
            },
            "completed": list(completed),
            "oom_retries": retries,
            "gpu_memory": {
                str(index): {
                    "total_mib": value.total_mib,
                    "free_mib": value.free_mib,
                    "allowed": index in self.allowed,
                }
                for index, value in sorted(memory.items())
            },
        }

    def run_workers(self, manifests: dict[str, Path]) -> None:
        name = "oracle_shapley_workers"
        self._set_phase(name, "running", heartbeat=utc_now())
        pending = self._tasks(manifests)
        completed: list[str] = []
        retries: dict[str, int] = {}
        quarantined_until: dict[int, float] = {}
        while pending or self.running:
            if self.stop_requested:
                raise InterruptedError("stop requested")
            now = time.time()
            for physical_gpu, running in list(self.running.items()):
                returncode = running.process.poll()
                if returncode is None:
                    continue
                running.log_handle.flush()
                running.log_handle.close()
                del self.running[physical_gpu]
                if returncode == 0:
                    completed.append(running.task.key)
                    self._log(f"completed {running.task.key} on physical_gpu={physical_gpu}")
                    continue
                oom = log_segment_has_cuda_oom(
                    running.task.log, running.log_start_offset
                )
                if oom and running.task.retries < self.retry_limit:
                    running.task.retries += 1
                    retries[running.task.key] = running.task.retries
                    running.task.not_before = now + self.retry_cooldown
                    quarantined_until[physical_gpu] = now + self.retry_cooldown
                    pending.append(running.task)
                    self._log(
                        f"CUDA OOM requeue {running.task.key} retry={running.task.retries}/"
                        f"{self.retry_limit}; physical_gpu={physical_gpu} cooldown="
                        f"{self.retry_cooldown}s"
                    )
                    continue
                kind = "OOM retries exhausted" if oom else "unhandled worker failure"
                raise RuntimeError(
                    f"{kind}: {running.task.key} exit={returncode} log={running.task.log}"
                )

            memory = query_gpu_memory()
            active_quarantine = {
                gpu for gpu, until in quarantined_until.items() if until > now
            }
            eligible = rank_eligible_gpus(
                memory,
                minimum_free_mib=self.minimum_free,
                allowed=self.allowed,
                busy=self.running,
                quarantined=active_quarantine,
            )
            pending.sort(key=lambda task: (task.not_before, task.shard_id, task.family))
            for physical_gpu in eligible:
                ready_index = next(
                    (index for index, task in enumerate(pending) if task.not_before <= now),
                    None,
                )
                if ready_index is None:
                    break
                task = pending.pop(ready_index)
                required = self.minimum_free + task.retries * self.retry_headroom
                if memory[physical_gpu].free_mib < required:
                    pending.append(task)
                    continue
                self.running[physical_gpu] = self._launch(task, physical_gpu)

            progress = self._worker_progress(
                pending=pending,
                completed=completed,
                memory=memory,
                retries=retries,
            )
            self._set_phase(name, "running", **progress)
            if pending or self.running:
                time.sleep(self.poll_seconds)
        self._set_phase(
            name,
            "passed",
            heartbeat=utc_now(),
            pending=[],
            running={},
            completed=completed,
            oom_retries=retries,
            allowed_physical_gpus=self.allowed,
            excluded_physical_gpus=self.excluded,
        )

    def combine(self, manifests: dict[str, Path]) -> dict[str, Any]:
        name = "combine_v01_statistics"
        self._set_phase(name, "running", heartbeat=utc_now())
        output_json = self.root / "artifacts/v01/statistics/v0_1_results.json"
        output_report = self.root / "artifacts/v01/reports/V0_1_final_report.md"
        command = [
            sys.executable,
            "-m",
            "rgrd.experiments.combine_v01",
            "--root",
            str(self.root),
            "--input-dir",
            str(self.root / "artifacts/v01/events"),
            "--output-json",
            str(output_json),
            "--output-report",
            str(output_report),
        ]
        for family, path in manifests.items():
            command.extend(["--manifest", f"{family}={path}"])
        self._run_cpu_command("v01-combine", command)
        result = json.loads(output_json.read_text(encoding="utf-8"))
        primary_status = GateStatus(result["gate_v01"]["status"])
        robustness_status = GateStatus(result["mask_direction_robustness"]["status"])

        def mutate(state: RunState) -> None:
            state.gates["gate_v01"] = GateDecision(
                gate="gate_v01",
                status=primary_status,
                evidence_paths=[str(output_json), str(output_report)],
                reasons=list(result["gate_v01"]["reasons"]),
            )
            state.gates["gate_v01_robustness"] = GateDecision(
                gate="gate_v01_robustness",
                status=robustness_status,
                evidence_paths=[str(output_json), str(output_report)],
                reasons=["attention-hole/donor direction agreement audit"],
            )

        self.store.mutate(mutate)
        self._set_phase(
            name,
            "passed",
            heartbeat=utc_now(),
            result_json=str(output_json),
            report=str(output_report),
        )
        return result

    def _terminate_children(self) -> None:
        for running in self.running.values():
            if running.process.poll() is None:
                running.process.terminate()
        deadline = time.time() + 20
        for running in self.running.values():
            timeout = max(0.0, deadline - time.time())
            try:
                running.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                running.process.kill()
            running.log_handle.close()

    def execute(self) -> int:
        self._archive_prior_state()
        self.store.save(self._initial_state())
        pid_path = self.root / "artifacts/runner.pid"
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        self._log(
            f"start run_id={self.run_id} commit={self.commit} "
            f"allowed={self.allowed} excluded={self.excluded}"
        )
        try:
            self.freeze_legacy()
            manifests = self.prepare_manifests()
            self.run_workers(manifests)
            result = self.combine(manifests)
            primary = GateStatus(result["gate_v01"]["status"])
            robustness = GateStatus(result["mask_direction_robustness"]["status"])

            def finish(state: RunState) -> None:
                state.current_phase = None
                state.status = (
                    "passed"
                    if primary == GateStatus.PASS and robustness == GateStatus.PASS
                    else "failed"
                )
                if state.status == "failed":
                    state.errors.append(
                        f"terminal gate status: primary={primary.value}, "
                        f"robustness={robustness.value}"
                    )

            self.store.mutate(finish)
            self._log(
                f"terminal primary={primary.value} robustness={robustness.value} "
                f"report={self.root / 'artifacts/v01/reports/V0_1_final_report.md'}"
            )
            science_passed = primary == GateStatus.PASS and robustness == GateStatus.PASS
            publication_succeeded = self._publish_terminal()
            if science_passed and publication_succeeded:
                return 0
            return 2 if not science_passed else 3
        except InterruptedError as exc:
            self._terminate_children()
            stop_message = str(exc)

            def stopped(state: RunState) -> None:
                state.status = "stopped"
                state.errors.append(stop_message)

            self.store.mutate(stopped)
            self._log("stopped by signal")
            self._publish_terminal()
            return 130
        except Exception as exc:
            self._terminate_children()
            self._fail(str(exc))
            self._publish_terminal()
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RGRD-V0.1 mechanism audit")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attack-root", type=Path, default=Path("/home/jkl/RAG_attack"))
    parser.add_argument(
        "--state", type=Path, default=Path("/mnt/data/jkl/RGRD_V0/artifacts/run_state.json")
    )
    parser.add_argument("--shards", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    runner = V01Runner(
        root=arguments.root,
        attack_root=arguments.attack_root,
        state_path=arguments.state,
        shards=arguments.shards,
    )

    def request_stop(_signum: int, _frame: object) -> None:
        runner.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return runner.execute()


if __name__ == "__main__":
    raise SystemExit(main())
