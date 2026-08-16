from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

import yaml

from rgrd.pipeline.state import GateDecision, PhaseRecord, RunState, StateStore
from rgrd.provenance import utc_now
from rgrd.publishing import publish_terminal_results
from rgrd.reporting import write_all, write_pipeline_report
from rgrd.experiments.gpu_admission import (
    log_segment_has_cuda_oom,
    query_gpu_memory,
    rank_eligible_gpus,
)
from rgrd.experiments.resume import event_provenance
from rgrd.experiments.scope import load_protocol_scope


PHASES = (
    "phase_0_provenance",
    "phase_1_attack_audit",
    "phase_2_deterministic_pipeline",
    "phase_3_span_intervention",
    "phase_4_role_attribution",
    "phase_5_oracle_mechanism",
    "phase_6_intervention_robustness",
    "phase_7_conformal_detection",
    "phase_8_hard_benign",
    "phase_9_joint_gcg_boundary",
)


class GateFailure(RuntimeError):
    def __init__(self, gate: str, reasons: Iterable[str]):
        self.gate = gate
        self.reasons = list(reasons)
        super().__init__(f"{gate} failed: {'; '.join(self.reasons)}")


@dataclass(frozen=True)
class ScheduledGpuJob:
    name: str
    command: list[str | Path]
    environment: dict[str, str] | None = None
    cwd: Path | None = None
    allowed: tuple[int, ...] = (0,)


@dataclass
class RunningGpuJob:
    job: ScheduledGpuJob
    process: subprocess.Popen[str]
    log: TextIO
    log_path: Path
    log_start_offset: int
    physical_gpu: int
    total_mib: int
    free_at_launch_mib: int
    attempt: int


class ExperimentRunner:
    def __init__(self, arguments: argparse.Namespace):
        self.root = arguments.root.resolve()
        self.attack_root = arguments.attack_root.resolve()
        self.state_path = arguments.state.resolve()
        self.log_path = arguments.log_path.resolve()
        self.python = Path(sys.executable).resolve()
        self.joint_python = arguments.joint_python.resolve()
        self.poll_seconds = arguments.poll_seconds
        self.store = StateStore(self.state_path)
        self.pipeline = yaml.safe_load(
            (self.root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8")
        )
        gpu_admission = self.pipeline.get("runtime", {}).get("gpu_admission", {})
        allowed_physical_gpus = gpu_admission.get("allowed_physical_gpus")
        if (
            not isinstance(allowed_physical_gpus, list)
            or not allowed_physical_gpus
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in allowed_physical_gpus
            )
            or len(allowed_physical_gpus) != len(set(allowed_physical_gpus))
        ):
            raise ValueError(
                "runtime.gpu_admission.allowed_physical_gpus must be a non-empty "
                "list of unique non-negative physical GPU indices"
            )
        self.gpu_allowed_physical_gpus = tuple(allowed_physical_gpus)
        self.gpu_full_min_free_mib = int(
            gpu_admission.get("full_model_min_free_mib", 19000)
        )
        self.gpu_index_min_free_mib = int(gpu_admission.get("index_min_free_mib", 2048))
        self.gpu_admission_poll_seconds = int(
            gpu_admission.get("poll_seconds", self.poll_seconds)
        )
        self.gpu_oom_retry_limit = int(gpu_admission.get("oom_retry_limit", 8))
        self.gpu_oom_retry_cooldown_seconds = int(
            gpu_admission.get("oom_retry_cooldown_seconds", 120)
        )
        self.gpu_oom_retry_headroom_mib = int(
            gpu_admission.get("oom_retry_headroom_mib", 1024)
        )
        positive_values = {
            "full_model_min_free_mib": self.gpu_full_min_free_mib,
            "index_min_free_mib": self.gpu_index_min_free_mib,
            "poll_seconds": self.gpu_admission_poll_seconds,
            "oom_retry_headroom_mib": self.gpu_oom_retry_headroom_mib,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid or self.gpu_oom_retry_limit < 0 or self.gpu_oom_retry_cooldown_seconds < 0:
            raise ValueError(f"invalid GPU admission configuration: {invalid}")
        self.datasets = yaml.safe_load(
            (self.root / "configs/datasets.yaml").read_text(encoding="utf-8")
        )["datasets"]
        self.preregistration = yaml.safe_load(
            (self.root / "configs/experiments/v0_preregistration.yaml").read_text(encoding="utf-8")
        )
        self.scope = load_protocol_scope(self.root)
        self.revisions = {
            role: str(model["revision"]) for role, model in self.pipeline["models"].items()
        }
        self.detector_commit = self._git("rev-parse", "HEAD")
        self.run_provenance = event_provenance(
            self.root,
            detector_code_commit=self.detector_commit,
            model_revisions=self.revisions,
        )
        self.base_environment = os.environ.copy()
        self.base_environment.update(
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": ",".join(
                    str(index) for index in self.gpu_allowed_physical_gpus
                ),
            }
        )
        self.state = self._new_state()

    def _git(self, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def _new_state(self) -> RunState:
        if self.state_path.is_file():
            try:
                previous = self.store.load()
            except Exception:
                previous = None
            if previous and previous.status == "running" and previous.pid:
                try:
                    os.kill(previous.pid, 0)
                except OSError:
                    pass
                else:
                    raise RuntimeError(f"another RGRD runner is alive with pid {previous.pid}")
            archive = self.state_path.with_name(
                f"{self.state_path.stem}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            )
            shutil.copy2(self.state_path, archive)
        run_id = "rgrd-v0-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return RunState(
            run_id=run_id,
            status="created",
            root=str(self.root),
            detector_code_commit=self.detector_commit,
            model_revisions=self.revisions,
            pipeline_config_sha256=self.run_provenance["pipeline_config_sha256"],
            pid=os.getpid(),
            log_path=str(self.log_path),
            phases={name: PhaseRecord(phase=name) for name in PHASES},
        )

    def _save(self) -> None:
        self.store.save(self.state)

    def _publish_terminal(self) -> bool:
        try:
            result = publish_terminal_results(self.root, self.state_path)
        except Exception as exc:
            message = f"terminal result publication failed: {type(exc).__name__}: {exc}"
            self.state.publication = {
                "status": "failed",
                "error": message,
                "at": utc_now(),
            }
            self.state.errors.append(message)
            self._save()
            print(f"[{utc_now()}] {message}", flush=True)
            return False
        self.state.publication = dict(result)
        self._save()
        print(
            f"[{utc_now()}] publication status={result['status']} "
            f"branch={result.get('branch')} commit={result.get('commit')} "
            f"destination={result.get('destination')}",
            flush=True,
        )
        return result["status"] in {"published", "already_published"}

    def _progress(self, **values: Any) -> None:
        if self.state.current_phase:
            self.state.phases[self.state.current_phase].progress.update(values)
        self._save()

    def _start_phase(self, name: str) -> None:
        record = self.state.phases[name]
        record.status = "running"
        record.started_at = utc_now()
        record.finished_at = None
        record.error = None
        self.state.current_phase = name
        self._save()

    def _finish_phase(self, name: str, progress: dict[str, Any] | None = None) -> None:
        record = self.state.phases[name]
        record.status = "passed"
        record.finished_at = utc_now()
        if progress:
            record.progress.update(progress)
        self.state.current_phase = None
        self._save()
        write_pipeline_report(self.root, self.state.model_dump(mode="json"))

    def _fail_phase(self, name: str, error: BaseException) -> None:
        record = self.state.phases[name]
        record.status = "failed"
        record.finished_at = utc_now()
        record.error = f"{type(error).__name__}: {error}"
        self.state.current_phase = name
        self._save()

    def _record_gate(
        self, gate: str, passed: bool, evidence: Iterable[Path], reasons: Iterable[str]
    ) -> None:
        self.state.gates[gate] = GateDecision(
            gate=gate,
            passed=passed,
            evidence_paths=[str(path.resolve()) for path in evidence],
            reasons=list(reasons),
        )
        self._save()

    def _log_file(self, label: str) -> Path:
        path = self.root / "artifacts/logs" / f"{label}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _run_command(
        self,
        label: str,
        command: list[str | Path],
        *,
        environment: dict[str, str] | None = None,
        cwd: Path | None = None,
        allowed: tuple[int, ...] = (0,),
    ) -> int:
        log_path = self._log_file(label)
        merged_environment = self.base_environment.copy()
        if environment:
            merged_environment.update(environment)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] COMMAND {json.dumps([str(item) for item in command])}\n")
            log.flush()
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=cwd or self.root,
                env=merged_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                self._progress(active_command=label, active_pid=process.pid, heartbeat=utc_now())
                time.sleep(self.poll_seconds)
            code = int(process.returncode)
            log.write(f"[{utc_now()}] EXIT {code}\n")
        if code not in allowed:
            raise RuntimeError(f"command {label} exited {code}; see {log_path}")
        return code

    def _run_parallel(
        self,
        label: str,
        jobs: list[tuple[str, list[str | Path], dict[str, str] | None, Path | None]],
    ) -> None:
        running: list[tuple[str, subprocess.Popen[str], Any, Path]] = []
        try:
            for job_name, command, environment, cwd in jobs:
                log_path = self._log_file(f"{label}-{job_name}")
                log = log_path.open("a", encoding="utf-8")
                log.write(
                    f"\n[{utc_now()}] COMMAND {json.dumps([str(item) for item in command])}\n"
                )
                log.flush()
                merged = self.base_environment.copy()
                if environment:
                    merged.update(environment)
                process = subprocess.Popen(
                    [str(item) for item in command],
                    cwd=cwd or self.root,
                    env=merged,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running.append((job_name, process, log, log_path))
            while True:
                statuses = {name: process.poll() for name, process, _, _ in running}
                self._progress(
                    active_group=label,
                    worker_status=statuses,
                    heartbeat=utc_now(),
                )
                failures = [name for name, code in statuses.items() if code not in (None, 0)]
                if failures:
                    raise RuntimeError(f"parallel group {label} failed workers: {failures}")
                if all(code == 0 for code in statuses.values()):
                    break
                time.sleep(self.poll_seconds)
        except BaseException:
            for _, process, _, _ in running:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            for _, process, _, _ in running:
                if process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
            raise
        finally:
            for name, process, log, _ in running:
                log.write(f"[{utc_now()}] EXIT {process.poll()}\n")
                log.close()

    def _run_gpu_queue(
        self,
        label: str,
        jobs: list[ScheduledGpuJob],
        *,
        minimum_free_mib: int,
    ) -> dict[str, int]:
        """Run GPU jobs only when admission headroom is available.

        Shards are independent of physical GPU indices. A CUDA OOM retries only the
        affected shard, with additional headroom and a temporary GPU quarantine.
        """

        if not jobs:
            return {}
        names = [job.name for job in jobs]
        if len(names) != len(set(names)):
            raise ValueError(f"GPU queue {label} has duplicate job names")
        pending = list(jobs)
        running: dict[str, RunningGpuJob] = {}
        completed: dict[str, int] = {}
        retries = {job.name: 0 for job in jobs}
        required_free = {job.name: minimum_free_mib for job in jobs}
        quarantined_until: dict[int, float] = {}
        single_job = len(jobs) == 1

        try:
            while pending or running:
                now = time.monotonic()
                for name, active in list(running.items()):
                    code = active.process.poll()
                    if code is None:
                        continue
                    code = int(code)
                    active.log.write(f"[{utc_now()}] EXIT {code}\n")
                    active.log.flush()
                    is_oom = code != 0 and log_segment_has_cuda_oom(
                        active.log_path, active.log_start_offset
                    )
                    active.log.close()
                    del running[name]
                    if is_oom and retries[name] < self.gpu_oom_retry_limit:
                        retries[name] += 1
                        maximum_safe_threshold = max(minimum_free_mib, active.total_mib - 512)
                        required_free[name] = min(
                            minimum_free_mib
                            + retries[name] * self.gpu_oom_retry_headroom_mib,
                            maximum_safe_threshold,
                        )
                        quarantined_until[active.physical_gpu] = (
                            now + self.gpu_oom_retry_cooldown_seconds
                        )
                        pending.append(active.job)
                        self._progress(
                            active_group=label,
                            last_oom_retry={
                                "job": name,
                                "physical_gpu": active.physical_gpu,
                                "attempt": active.attempt,
                                "retry": retries[name],
                                "next_min_free_mib": required_free[name],
                                "log": str(active.log_path),
                                "at": utc_now(),
                            },
                            heartbeat=utc_now(),
                        )
                    elif code in active.job.allowed and not is_oom:
                        completed[name] = code
                    else:
                        failure_kind = "CUDA OOM retry limit exhausted" if is_oom else "failure"
                        raise RuntimeError(
                            f"GPU queue {label} job {name} {failure_kind} with exit {code}; "
                            f"see {active.log_path}"
                        )

                memory = query_gpu_memory()
                missing_allowed = set(self.gpu_allowed_physical_gpus) - set(memory)
                if missing_allowed:
                    raise RuntimeError(
                        "configured physical GPUs are not reported by nvidia-smi: "
                        f"{sorted(missing_allowed)}"
                    )
                allowed_memory = {
                    index: memory[index] for index in self.gpu_allowed_physical_gpus
                }
                if minimum_free_mib > max(
                    value.total_mib for value in allowed_memory.values()
                ):
                    raise RuntimeError(
                        f"GPU queue {label} requires {minimum_free_mib} MiB free, exceeding "
                        "all allowed GPU capacities"
                    )
                quarantined = {
                    index for index, until in quarantined_until.items() if until > now
                }
                busy = {active.physical_gpu for active in running.values()}

                while pending:
                    selected_job_index: int | None = None
                    selected_gpu: int | None = None
                    for job_index, job in enumerate(pending):
                        candidates = rank_eligible_gpus(
                            memory,
                            minimum_free_mib=required_free[job.name],
                            allowed=self.gpu_allowed_physical_gpus,
                            busy=busy,
                            quarantined=quarantined,
                        )
                        if candidates:
                            selected_job_index = job_index
                            selected_gpu = candidates[0]
                            break
                    if selected_job_index is None or selected_gpu is None:
                        break
                    job = pending.pop(selected_job_index)
                    gpu_memory = memory[selected_gpu]
                    log_path = self._log_file(label if single_job else f"{label}-{job.name}")
                    log = log_path.open("a", encoding="utf-8")
                    attempt = retries[job.name] + 1
                    log.write(
                        f"\n[{utc_now()}] GPU_ADMISSION physical_gpu={selected_gpu} "
                        f"free_mib={gpu_memory.free_mib} required_mib={required_free[job.name]} "
                        f"attempt={attempt}\n"
                    )
                    log.write(
                        f"[{utc_now()}] COMMAND "
                        f"{json.dumps([str(item) for item in job.command])}\n"
                    )
                    log.flush()
                    log_start_offset = log.tell()
                    environment = self.base_environment.copy()
                    if job.environment:
                        environment.update(job.environment)
                    environment["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
                    process = subprocess.Popen(
                        [str(item) for item in job.command],
                        cwd=job.cwd or self.root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    running[job.name] = RunningGpuJob(
                        job=job,
                        process=process,
                        log=log,
                        log_path=log_path,
                        log_start_offset=log_start_offset,
                        physical_gpu=selected_gpu,
                        total_mib=gpu_memory.total_mib,
                        free_at_launch_mib=gpu_memory.free_mib,
                        attempt=attempt,
                    )
                    busy.add(selected_gpu)

                self._progress(
                    active_group=label,
                    gpu_admission={
                        "allowed_physical_gpus": list(self.gpu_allowed_physical_gpus),
                        "excluded_physical_gpus": sorted(
                            set(memory) - set(self.gpu_allowed_physical_gpus)
                        ),
                        "base_min_free_mib": minimum_free_mib,
                        "pending": [job.name for job in pending],
                        "running": {
                            name: {
                                "pid": active.process.pid,
                                "physical_gpu": active.physical_gpu,
                                "attempt": active.attempt,
                                "free_at_launch_mib": active.free_at_launch_mib,
                            }
                            for name, active in sorted(running.items())
                        },
                        "completed": sorted(completed),
                        "oom_retries": retries,
                        "required_free_mib": required_free,
                        "gpu_memory_mib": {
                            str(index): {
                                "total": value.total_mib,
                                "free": value.free_mib,
                            }
                            for index, value in sorted(memory.items())
                        },
                        "quarantined_gpus": sorted(quarantined),
                    },
                    heartbeat=utc_now(),
                )
                if pending or running:
                    time.sleep(self.gpu_admission_poll_seconds)
        except BaseException:
            for active in running.values():
                if active.process.poll() is None:
                    active.process.send_signal(signal.SIGTERM)
            for active in running.values():
                if active.process.poll() is None:
                    try:
                        active.process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        active.process.kill()
            raise
        finally:
            for active in running.values():
                if not active.log.closed:
                    active.log.write(f"[{utc_now()}] EXIT {active.process.poll()}\n")
                    active.log.close()
        return completed

    def _run_gpu_command(
        self,
        label: str,
        command: list[str | Path],
        *,
        environment: dict[str, str] | None = None,
        cwd: Path | None = None,
        allowed: tuple[int, ...] = (0,),
        minimum_free_mib: int | None = None,
    ) -> int:
        results = self._run_gpu_queue(
            label,
            [
                ScheduledGpuJob(
                    name="single",
                    command=command,
                    environment=environment,
                    cwd=cwd,
                    allowed=allowed,
                )
            ],
            minimum_free_mib=minimum_free_mib or self.gpu_full_min_free_mib,
        )
        return results["single"]

    def _model_revision_arguments(self) -> list[str]:
        arguments: list[str] = []
        for role, revision in sorted(self.revisions.items()):
            arguments.extend(["--model-revision", f"{role}={revision}"])
        return arguments

    def _validate_runtime_locks(self) -> dict[str, Any]:
        runtime_upstreams = self.root / "artifacts/provenance/upstreams.runtime.yaml"
        runtime_models = self.root / "artifacts/provenance/models.runtime.yaml"
        self._run_command(
            "phase0-pin-upstreams",
            [
                self.python,
                self.root / "scripts/pin_upstreams.py",
                "--root",
                self.root / "third_party",
                "--config",
                self.root / "configs/upstream_projects.yaml",
                "--output",
                runtime_upstreams,
            ],
        )
        self._run_command(
            "phase0-pin-models",
            [
                self.python,
                self.root / "scripts/pin_models.py",
                "--sources",
                self.root / "configs/model_sources.yaml",
                "--pipeline",
                self.root / "configs/pipeline/v0.yaml",
                "--output",
                runtime_models,
            ],
        )
        expected_upstreams = yaml.safe_load(
            (self.root / "third_party/manifest.lock.yaml").read_text(encoding="utf-8")
        )
        current_upstreams = yaml.safe_load(runtime_upstreams.read_text(encoding="utf-8"))
        expected_projects = {item["name"]: item for item in expected_upstreams["projects"]}
        current_projects = {item["name"]: item for item in current_upstreams["projects"]}
        if set(expected_projects) != set(current_projects):
            raise RuntimeError("upstream lock project set differs from runtime clones")
        for name in expected_projects:
            if expected_projects[name]["commit"] != current_projects[name]["commit"]:
                raise RuntimeError(f"upstream commit changed for {name}")
            if current_projects[name]["local_modifications"]["dirty"]:
                raise RuntimeError(f"pinned upstream clone is dirty: {name}")
        expected_models = yaml.safe_load(
            (self.root / "configs/models.lock.yaml").read_text(encoding="utf-8")
        )
        current_models = yaml.safe_load(runtime_models.read_text(encoding="utf-8"))
        for role, expected in expected_models["models"].items():
            current = current_models["models"].get(role)
            if not current or expected["revision"] != current["revision"]:
                raise RuntimeError(f"model revision changed for {role}")
            if expected["content_sha256"] != current["content_sha256"]:
                raise RuntimeError(f"model content hash changed for {role}")
        return {
            "upstream_runtime_lock": str(runtime_upstreams),
            "model_runtime_lock": str(runtime_models),
        }

    def _validate_project_tree(self) -> None:
        status = self._git("status", "--porcelain", "--untracked-files=all")
        unexpected: list[str] = []
        for line in status.splitlines():
            path = line[3:]
            if " -> " in path:
                path = path.rsplit(" -> ", 1)[1]
            path = path.strip('"').replace("\\", "/")
            if not path.startswith("artifacts/"):
                unexpected.append(line)
        if unexpected:
            raise RuntimeError(
                "RGRD source tree has uncommitted non-artifact changes: "
                + "; ".join(unexpected[:20])
            )

    def _capture_environment(self) -> Path:
        output = self.root / "artifacts/provenance/rgrd_environment.json"
        completed = subprocess.run(
            [str(self.python), "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
            env=self.base_environment,
        )
        conda = shutil.which("conda")
        explicit: list[str] = []
        if conda:
            result = subprocess.run(
                [conda, "list", "--explicit", "--prefix", str(self.python.parent.parent)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                explicit = result.stdout.splitlines()
        value = {
            "schema_version": 1,
            "captured_at": utc_now(),
            "python": str(self.python),
            "pip_freeze": completed.stdout.splitlines(),
            "conda_explicit": explicit,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def _capture_joint_environment(self) -> Path:
        if not self.joint_python.is_file():
            raise FileNotFoundError(
                f"separate Joint-GCG environment is absent: {self.joint_python}"
            )
        output = self.root / "artifacts/provenance/joint_gcg_environment.json"
        completed = subprocess.run(
            [str(self.joint_python), "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
            env={**self.base_environment, "PYTHONPATH": str(self.root)},
        )
        probe = subprocess.run(
            [
                str(self.joint_python),
                "-c",
                (
                    "import json,numpy,torch,transformers,lightning; "
                    "print(json.dumps({'numpy':numpy.__version__,'torch':torch.__version__,"
                    "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
                    "'transformers':transformers.__version__,'lightning':lightning.__version__},"
                    "sort_keys=True))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**self.base_environment, "PYTHONPATH": str(self.root)},
        )
        value = {
            "schema_version": 1,
            "captured_at": utc_now(),
            "python": str(self.joint_python),
            "versions": json.loads(probe.stdout.strip().splitlines()[-1]),
            "pip_freeze": completed.stdout.splitlines(),
            "environment_declaration": str((self.root / "envs/upstream-joint-gcg.yml").resolve()),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def _phase_0(self) -> dict[str, Any]:
        self._validate_project_tree()
        if (
            self.attack_root == self.root
            or self.root in self.attack_root.parents
            or self.attack_root in self.root.parents
        ):
            raise RuntimeError("RGRD output root overlaps the read-only attack root")
        if not self.attack_root.is_dir():
            raise FileNotFoundError(self.attack_root)
        environments = yaml.safe_load(
            (self.root / "configs/upstream_projects.yaml").read_text(encoding="utf-8")
        )["projects"]
        declarations = [value["environment"] for value in environments.values()]
        if len(declarations) != len(set(declarations)):
            raise RuntimeError("paper projects do not have unique environment declarations")
        locks = self._validate_runtime_locks()
        environment = self._capture_environment()
        joint_environment = self._capture_joint_environment()
        self._run_command(
            "phase0-pytest",
            [self.python, "-m", "pytest", "-q"],
        )
        self._run_gpu_command(
            "phase0-model-preflight",
            [
                self.python,
                "-m",
                "rgrd.experiments.model_preflight",
                "--root",
                self.root,
                "--device",
                "cuda:0",
            ],
        )
        return {
            **locks,
            "environment": str(environment),
            "joint_environment": str(joint_environment),
            "tests": "passed",
            "models": "passed",
            "protocol_scope": self.scope.metadata(),
        }

    def _phase_1(self) -> dict[str, Any]:
        whitebox = self.root / "artifacts/attacks/poisonedrag_w_nq100.json"
        self._run_gpu_command(
            "phase1-poisonedrag-whitebox-smoke",
            [
                self.python,
                "-m",
                "rgrd.attacks.poisonedrag_whitebox",
                "--attack-root",
                self.attack_root,
                "--retriever",
                self.pipeline["models"]["retriever"]["local_path"],
                "--generator",
                self.pipeline["models"]["generator"]["local_path"],
                "--output",
                whitebox,
                "--count",
                "20",
                "--gpu",
                "0",
                "--seed",
                "12",
            ],
        )
        audit_dir = self.root / "artifacts/audit"
        self._run_command(
            "phase1-reproduction-audit",
            [
                self.python,
                "-m",
                "rgrd.audit.scan",
                "--root",
                self.attack_root,
                "--upstream-manifest",
                self.root / "configs/upstreams.yaml",
                "--output",
                audit_dir,
            ],
            allowed=(0, 2),
        )
        result = json.loads((audit_dir / "reproduction_audit.json").read_text(encoding="utf-8"))
        passed = result["gate_1"]["status"] == "PASS"
        reasons = list(result["gate_1"]["reasons"])
        self._record_gate(
            "gate_1",
            passed,
            [audit_dir / "reproduction_audit.json", audit_dir / "reproduction_audit.md"],
            reasons,
        )
        if not passed:
            raise GateFailure("gate_1", reasons)
        return {"whitebox_smoke": 20, "audit": "PASS"}

    def _index_progress(self, output: Path) -> dict[str, Any]:
        values = []
        for path in sorted(output.glob("shard-*.progress.json")):
            try:
                values.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return {
            "shards_reporting": len(values),
            "documents": sum(int(value.get("documents", 0)) for value in values),
            "chunks": sum(int(value.get("chunks", 0)) for value in values),
        }

    def _build_index(self, dataset: str) -> dict[str, Any]:
        output = self.root / f"artifacts/indexes/{dataset}"
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("index_type") == "IndexFlatIP"
                and Path(manifest["index_path"]).is_file()
                and Path(manifest["database_path"]).is_file()
            ):
                return manifest
        jobs: list[ScheduledGpuJob] = []
        for shard_id in range(8):
            jobs.append(
                ScheduledGpuJob(
                    name=f"shard-{shard_id:02d}",
                    command=[
                        self.python,
                        "-m",
                        "rgrd.indexing.exact",
                        "shard",
                        "--corpus",
                        self.datasets[dataset]["corpus"],
                        "--generator-tokenizer",
                        self.pipeline["models"]["generator"]["local_path"],
                        "--retriever",
                        self.pipeline["models"]["retriever"]["local_path"],
                        "--output",
                        output,
                        "--shard-id",
                        str(shard_id),
                        "--shards",
                        "8",
                        "--device",
                        "cuda:0",
                    ],
                    cwd=self.root,
                )
            )
        self._run_gpu_queue(
            f"index-{dataset}",
            jobs,
            minimum_free_mib=self.gpu_index_min_free_mib,
        )
        self._progress(index_dataset=dataset, index_progress=self._index_progress(output))
        self._run_command(
            f"index-{dataset}-merge",
            [
                self.python,
                "-m",
                "rgrd.indexing.exact",
                "merge",
                "--output",
                output,
                "--shards",
                "8",
            ],
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["index_type"] != "IndexFlatIP" or int(manifest["chunks"]) <= 0:
            raise RuntimeError(f"invalid exact index for {dataset}")
        return manifest

    def _phase_2(self) -> dict[str, Any]:
        index = self._build_index("nq")
        self._run_command(
            "phase2-pytest",
            [self.python, "-m", "pytest", "-q"],
        )
        self._run_gpu_command(
            "phase2-determinism",
            [
                self.python,
                "-m",
                "rgrd.experiments.gates",
                "determinism",
                "--root",
                self.root,
                "--dataset",
                "nq",
                "--count",
                "20",
                "--device",
                "cuda:0",
                "--detector-commit",
                self.detector_commit,
            ],
            allowed=(0, 2),
        )
        path = self.root / "artifacts/statistics/gate2_determinism.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        passed = bool(result["gate_2"]["passed"])
        reasons = list(result["gate_2"]["reasons"])
        self._record_gate("gate_2", passed, [path], reasons)
        if not passed:
            raise GateFailure("gate_2", reasons)
        return {"nq_index_chunks": int(index["chunks"]), "determinism_queries": 20}

    def _phase_3(self) -> dict[str, Any]:
        self._run_gpu_command(
            "phase3-noop-noise",
            [
                self.python,
                "-m",
                "rgrd.experiments.gates",
                "noop",
                "--root",
                self.root,
                "--dataset",
                "nq",
                "--count",
                "20",
                "--device",
                "cuda:0",
                "--detector-commit",
                self.detector_commit,
            ],
        )
        result = json.loads(
            (self.root / "artifacts/statistics/noop_noise_floor.json").read_text(encoding="utf-8")
        )
        return {
            "observations": result["observations"],
            "retrieval_floor": result["retrieval"],
            "generation_floor": result["generation"],
        }

    def _phase_4(self) -> dict[str, Any]:
        count = int(self.preregistration["e01_determinism"]["attribution_pilot_queries"])
        self._run_gpu_command(
            "phase4-attribution-pilot",
            [
                self.python,
                "-m",
                "rgrd.experiments.gates",
                "pilot",
                "--root",
                self.root,
                "--dataset",
                "nq",
                "--count",
                str(count),
                "--device",
                "cuda:0",
                "--detector-commit",
                self.detector_commit,
            ],
        )
        return {"pilot_queries": count, "runtime_view": "label-free"}

    def _whitebox_full(self) -> None:
        self._run_gpu_command(
            "phase5-poisonedrag-whitebox-100",
            [
                self.python,
                "-m",
                "rgrd.attacks.poisonedrag_whitebox",
                "--attack-root",
                self.attack_root,
                "--retriever",
                self.pipeline["models"]["retriever"]["local_path"],
                "--generator",
                self.pipeline["models"]["generator"]["local_path"],
                "--output",
                self.root / "artifacts/attacks/poisonedrag_w_nq100.json",
                "--count",
                "100",
                "--gpu",
                "0",
                "--seed",
                "12",
            ],
        )

    def _worker_jobs(
        self,
        *,
        module: str,
        output_dir: Path,
        common: list[str | Path],
        label: str,
        python: Path | None = None,
        cwd: Path | None = None,
    ) -> None:
        jobs: list[ScheduledGpuJob] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for shard_id in range(8):
            command = [
                python or self.python,
                "-m",
                module,
                *common,
                "--output",
                output_dir / f"shard-{shard_id:02d}.jsonl",
                "--shard-id",
                str(shard_id),
                "--shards",
                "8",
                "--device",
                "cuda:0",
                "--detector-commit",
                self.detector_commit,
                *self._model_revision_arguments(),
            ]
            jobs.append(
                ScheduledGpuJob(
                    name=f"shard-{shard_id:02d}",
                    command=command,
                    cwd=cwd,
                )
            )
        self._run_gpu_queue(
            label,
            jobs,
            minimum_free_mib=self.gpu_full_min_free_mib,
        )

    def _phase_5(self) -> dict[str, Any]:
        families = self.scope.family_datasets
        index_stats = {
            dataset: self._build_index(dataset)
            for dataset in sorted({dataset for _, dataset in families})
        }
        self._whitebox_full()
        output = self.root / "artifacts/events/e02_mechanism"
        for family, dataset in families:
            safe = family.lower().replace("-", "_")
            self._worker_jobs(
                module="rgrd.experiments.mechanism_worker",
                output_dir=output / safe,
                common=[
                    "--root",
                    self.root,
                    "--attack-root",
                    self.attack_root,
                    "--family",
                    family,
                    "--dataset",
                    dataset,
                    "--max-candidates",
                    "500",
                ],
                label=f"mechanism-{safe}",
            )
        combined_input = self.root / "artifacts/events/e02_mechanism_combined"
        combined_input.mkdir(parents=True, exist_ok=True)
        for family, _ in families:
            safe = family.lower().replace("-", "_")
            for path in (output / safe).glob("*.jsonl"):
                target = combined_input / f"{path.parent.name}-{path.name}"
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(path.resolve())
        self._run_command(
            "phase5-combine-mechanism",
            [
                self.python,
                "-m",
                "rgrd.experiments.combine_mechanism",
                "--root",
                self.root,
                "--input",
                combined_input,
                "--seed",
                str(self.pipeline["seed"]),
            ],
            allowed=(0, 2),
        )
        stats = self.root / "artifacts/statistics/mechanism_tests.json"
        result = json.loads(stats.read_text(encoding="utf-8"))
        gate = result["gate_3"]
        passed = bool(gate["passed"])
        reasons = list(gate.get("reasons", [gate.get("reason", "Gate 3 failed")]))
        self._record_gate(
            "gate_3",
            passed,
            [stats, self.root / "artifacts/reports/mechanism_report.md"],
            reasons,
        )
        if not passed:
            raise GateFailure("gate_3", reasons)
        return {
            "index_chunks": {
                dataset: int(statistics["chunks"]) for dataset, statistics in index_stats.items()
            },
            "mechanism_observations": result["observations"],
            "protocol_scope": self.scope.metadata(),
        }

    def _phase_6(self) -> dict[str, Any]:
        mechanism = self.root / "artifacts/events/e02_mechanism_combined"
        output = self.root / "artifacts/events/e03_robustness"
        active_datasets = sorted({dataset for _, dataset in self.scope.family_datasets})
        for dataset in active_datasets:
            self._worker_jobs(
                module="rgrd.experiments.robustness_worker",
                output_dir=output / dataset,
                common=[
                    "--root",
                    self.root,
                    "--attack-root",
                    self.attack_root,
                    "--mechanism-input",
                    mechanism,
                    "--dataset",
                    dataset,
                ],
                label=f"robustness-{dataset}",
            )
        combined = self.root / "artifacts/events/e03_robustness_combined"
        combined.mkdir(parents=True, exist_ok=True)
        for dataset in active_datasets:
            for path in (output / dataset).glob("*.jsonl"):
                target = combined / f"{path.parent.name}-{path.name}"
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(path.resolve())
        self._run_command(
            "phase6-combine-robustness",
            [
                self.python,
                "-m",
                "rgrd.experiments.combine_robustness",
                "--root",
                self.root,
                "--input",
                combined,
            ],
            allowed=(0, 2),
        )
        stats = self.root / "artifacts/statistics/intervention_robustness.json"
        result = json.loads(stats.read_text(encoding="utf-8"))
        gate = result["gate_robustness"]
        passed = bool(gate["passed"])
        reasons = [
            gate["reason"],
            *[
                f"{role}: agreement={value['agreement_rate']:.6f}, ci_lower={value['exact_95_ci'][0]:.6f}"
                for role, value in result["groups"].items()
            ],
        ]
        self._record_gate("gate_robustness", passed, [stats], reasons)
        if not passed:
            raise GateFailure("gate_robustness", reasons)
        return {"groups": result["groups"]}

    def _detection_group(
        self,
        *,
        label: str,
        mode: str,
        partition: str,
        dataset: str,
        start: int,
        limit: int,
        family: str = "clean",
    ) -> None:
        output = self.root / "artifacts/events/e04_detection" / label
        common: list[str | Path] = [
            "--root",
            self.root,
            "--attack-root",
            self.attack_root,
            "--mode",
            mode,
            "--partition",
            partition,
            "--family",
            family,
            "--dataset",
            dataset,
            "--start",
            str(start),
            "--limit",
            str(limit),
        ]
        if mode == "attack":
            common.extend(
                [
                    "--mechanism-input",
                    self.root / "artifacts/events/e02_mechanism_combined",
                ]
            )
        self._worker_jobs(
            module="rgrd.experiments.detection_worker",
            output_dir=output,
            common=common,
            label=f"detection-{label}",
        )

    def _phase_7(self) -> dict[str, Any]:
        self._detection_group(
            label="calibration",
            mode="clean",
            partition="calibration",
            dataset="nq",
            start=0,
            limit=200,
        )
        self._detection_group(
            label="clean_test",
            mode="clean",
            partition="clean_test",
            dataset="nq",
            start=200,
            limit=100,
        )
        for family, dataset in self.scope.family_datasets:
            self._detection_group(
                label=family.lower().replace("-", "_"),
                mode="attack",
                partition="attack_test",
                dataset=dataset,
                start=0,
                limit=50,
                family=family,
            )
        combined = self.root / "artifacts/events/e04_detection_combined"
        combined.mkdir(parents=True, exist_ok=True)
        active_labels = [
            "calibration",
            "clean_test",
            *(family.lower().replace("-", "_") for family in self.scope.active_attack_families),
        ]
        detection_root = self.root / "artifacts/events/e04_detection"
        for label in active_labels:
            for path in (detection_root / label).glob("*.jsonl"):
                target = combined / f"{path.parent.name}-{path.name}"
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(path.resolve())
        self._run_command(
            "phase7-combine-detection",
            [
                self.python,
                "-m",
                "rgrd.experiments.combine_detection",
                "--root",
                self.root,
                "--input",
                combined,
            ],
        )
        result = json.loads(
            (self.root / "artifacts/statistics/detection.json").read_text(encoding="utf-8")
        )
        return {
            "calibration_n": result["calibration_n"],
            "ordinary_clean_fpr": result["ordinary_clean"]["rate"],
            "families": {key: value["rate"] for key, value in result["per_family"].items()},
        }

    def _phase_8(self) -> dict[str, Any]:
        hotpot = self._build_index("hotpotqa")
        tasks = self.root / "artifacts/events/e05_hard_benign/tasks.jsonl"
        self._run_command(
            "phase8-build-hard-benign",
            [
                self.python,
                "-m",
                "rgrd.experiments.hard_benign",
                "build",
                "--root",
                self.root,
                "--output",
                tasks,
                "--per-group",
                "100",
            ],
        )
        output = self.root / "artifacts/events/e05_hard_benign/results"
        for dataset in ("nq", "hotpotqa"):
            self._worker_jobs(
                module="rgrd.experiments.hard_benign_worker",
                output_dir=output / dataset,
                common=[
                    "--root",
                    self.root,
                    "--tasks",
                    tasks,
                    "--dataset",
                    dataset,
                ],
                label=f"hard-benign-{dataset}",
            )
        combined = self.root / "artifacts/events/e05_hard_benign_combined"
        combined.mkdir(parents=True, exist_ok=True)
        for path in output.rglob("*.jsonl"):
            target = combined / f"{path.parent.name}-{path.name}"
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(path.resolve())
        self._run_command(
            "phase8-combine-hard-benign",
            [
                self.python,
                "-m",
                "rgrd.experiments.hard_benign",
                "combine",
                "--root",
                self.root,
                "--input",
                combined,
                "--detection-input",
                self.root / "artifacts/events/e04_detection_combined",
            ],
        )
        result = json.loads(
            (self.root / "artifacts/statistics/hard_benign.json").read_text(encoding="utf-8")
        )
        return {"hotpot_index_chunks": hotpot["chunks"], "groups": result["groups"]}

    def _phase_9(self) -> dict[str, Any]:
        config = self.preregistration["e06_joint_gcg"]
        official = self.root / "third_party/Joint-GCG"
        projection = self.root / "artifacts/joint_gcg/projection"
        if not self.joint_python.is_file():
            raise FileNotFoundError(
                f"separate Joint-GCG environment is absent: {self.joint_python}"
            )
        self._run_gpu_command(
            "phase9-joint-projection",
            [
                self.joint_python,
                "-m",
                "rgrd.attacks.joint_projection",
                "--retriever",
                self.pipeline["models"]["retriever"]["local_path"],
                "--generator",
                self.pipeline["models"]["generator"]["local_path"],
                "--output",
                projection,
                "--official-source",
                official / "model_projection/train_autoencoder.py",
                "--device",
                "cuda:0",
                "--seed",
                "42",
            ],
            environment={"PYTHONPATH": str(self.root)},
        )
        task_root = self.root / "artifacts/joint_gcg/tasks"
        self._run_command(
            "phase9-prepare-tasks",
            [
                self.python,
                "-m",
                "rgrd.attacks.joint_gcg",
                "prepare",
                "--official-root",
                official,
                "--output",
                task_root,
                "--count",
                str(config["attack_attempts"]),
                "--shards",
                str(config["shards"]),
            ],
        )
        logs = self.root / "artifacts/joint_gcg/official_logs"
        jobs: list[ScheduledGpuJob] = []
        for shard_id in range(8):
            command: list[str | Path] = [
                self.joint_python,
                "-m",
                "rgrd.attacks.joint_gcg",
                "worker",
                "--official-root",
                official,
                "--assets",
                task_root / f"shard-{shard_id:02d}",
                "--output",
                logs / f"shard-{shard_id:02d}",
                "--retriever",
                self.pipeline["models"]["retriever"]["local_path"],
                "--generator",
                self.pipeline["models"]["generator"]["local_path"],
                "--transfer-matrix",
                projection / "transfer_matrix.npy",
                "--corpus",
                self.datasets["nq"]["corpus"],
                "--retrieval-results",
                official / "attack_data/poisoned_rag_nq/nq_contriever_ground-truth.json",
                "--epochs",
                str(config["epochs"]),
                "--n-samples",
                str(config["candidates_per_step"]),
                "--topk",
                str(config["topk_per_coordinate"]),
                "--tag-length",
                str(config["tag_length"]),
            ]
            jobs.append(
                ScheduledGpuJob(
                    name=f"shard-{shard_id:02d}",
                    command=command,
                    environment={"PYTHONPATH": str(self.root)},
                    cwd=official,
                )
            )
        self._run_gpu_queue(
            "joint-gcg-official",
            jobs,
            minimum_free_mib=self.gpu_full_min_free_mib,
        )
        collected = self.root / "artifacts/attacks/joint_gcg_nq50.jsonl"
        self._run_command(
            "phase9-collect-joint",
            [
                self.python,
                "-m",
                "rgrd.attacks.joint_gcg",
                "collect",
                "--tasks-root",
                task_root,
                "--logs-root",
                logs,
                "--output",
                collected,
            ],
        )
        joint_results = self.root / "artifacts/events/e06_joint_gcg"
        self._worker_jobs(
            module="rgrd.experiments.joint_detection_worker",
            output_dir=joint_results,
            common=["--root", self.root, "--tasks", collected],
            label="joint-gcg-detection",
        )
        self._run_command(
            "phase9-combine-joint",
            [
                self.python,
                "-m",
                "rgrd.experiments.combine_joint_gcg",
                "--root",
                self.root,
                "--joint-input",
                joint_results,
                "--detection-input",
                self.root / "artifacts/events/e04_detection_combined",
                "--required",
                str(config["attack_attempts"]),
            ],
        )
        result = json.loads(
            (self.root / "artifacts/statistics/joint_gcg_boundary.json").read_text(encoding="utf-8")
        )
        return {
            "attempts": result["attempts"],
            "tpr_at_5pct": result["detection_at_query_fpr_5pct"]["rate"],
            "judgments": result["judgments"],
        }

    def run(self) -> int:
        self.state.status = "running"
        self.state.pid = os.getpid()
        self._save()
        implementations: list[tuple[str, Callable[[], dict[str, Any]]]] = [
            ("phase_0_provenance", self._phase_0),
            ("phase_1_attack_audit", self._phase_1),
            ("phase_2_deterministic_pipeline", self._phase_2),
            ("phase_3_span_intervention", self._phase_3),
            ("phase_4_role_attribution", self._phase_4),
            ("phase_5_oracle_mechanism", self._phase_5),
            ("phase_6_intervention_robustness", self._phase_6),
            ("phase_7_conformal_detection", self._phase_7),
            ("phase_8_hard_benign", self._phase_8),
            ("phase_9_joint_gcg_boundary", self._phase_9),
        ]
        try:
            for name, function in implementations:
                self._start_phase(name)
                try:
                    progress = function()
                except BaseException as exc:
                    self._fail_phase(name, exc)
                    raise
                self._finish_phase(name, progress)
            self.state.status = "passed"
            self.state.current_phase = None
            self._save()
            write_all(self.root, self.state.model_dump(mode="json"))
            return 0 if self._publish_terminal() else 3
        except BaseException as exc:
            self.state.status = "failed"
            self.state.errors.append(f"{type(exc).__name__}: {exc}")
            self.state.errors.append(traceback.format_exc())
            for phase in self.state.phases.values():
                if phase.status == "pending":
                    phase.status = "skipped"
                    phase.error = "not run because a preceding gate or phase failed"
            self.state.current_phase = None
            self._save()
            write_all(self.root, self.state.model_dump(mode="json"))
            self._publish_terminal()
            return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the gate-controlled RGRD-V0 experiment")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attack-root", type=Path, default=Path("/home/jkl/RAG_attack"))
    parser.add_argument("--state", type=Path, default=Path("artifacts/run_state.json"))
    parser.add_argument("--log-path", type=Path, default=Path("artifacts/logs/runner.log"))
    parser.add_argument(
        "--joint-python",
        type=Path,
        default=Path("/mnt/data/jkl/conda-envs/rgrd-v0-joint-gcg/bin/python"),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if not arguments.state.is_absolute():
        arguments.state = arguments.root / arguments.state
    if not arguments.log_path.is_absolute():
        arguments.log_path = arguments.root / arguments.log_path
    runner = ExperimentRunner(arguments)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
