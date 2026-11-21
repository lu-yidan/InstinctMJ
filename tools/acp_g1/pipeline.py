#!/usr/bin/env python3
"""Resume an ACP simulation training pipeline; hardware control is never started.

Default invocation prints the plan. --execute advances it, --once makes one
bounded scheduling pass. Every child has an external exit-status receipt, PID,
and log. Evaluation gates use SUCCESS averaged by original source motion hash,
with held-out source files kept separate from training files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


THRESHOLDS = {"train.paper": .9, "train.robust": .8, "holdout.paper": .8, "holdout.robust": .7}
TASKS = {1: "Instinct-ACP-Track-G1-v1", 2: "Instinct-ACP-Stage2-G1-v1", 3: "Instinct-ACP-Stage3-Flat-G1-v1"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def birth(pid):
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().split(")", 1)[1].split()[19]
    except (OSError, ValueError, TypeError, IndexError):
        return None


def alive(pid, started):
    return bool(started and birth(pid) == started)


def worker(spec_path):
    spec = json_read(spec_path)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in spec.get("env", {}).items()})
    log = Path(spec["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("ab", buffering=0) as handle:
            process = subprocess.Popen(spec["command"], cwd=spec["cwd"], env=env, stdout=handle, stderr=subprocess.STDOUT)
            json_write(spec["process_file"], {"pid": process.pid, "birth": birth(process.pid), "started_at": now(),
                                               "command": spec["command"], "log": str(log)})
            returncode = process.wait()
        json_write(spec["exit_file"], {"returncode": returncode, "finished_at": now(), "command": spec["command"]})
        return returncode
    except Exception as error:
        json_write(spec["exit_file"], {"returncode": 127, "finished_at": now(), "error": str(error), "command": spec["command"]})
        return 127


def checkpoint_receipt(path, stage, iterations):
    import torch
    path = Path(path).resolve(strict=True)
    sidecar = path.with_name("acp_state_" + path.name.removeprefix("model_"))
    if not sidecar.is_file():
        raise ValueError(f"Missing world-model/critic/planner training sidecar: {sidecar}")
    for name in ("agent.yaml", "env.yaml"):
        if not (path.parent / "params" / name).is_file():
            raise ValueError(f"Saved training configuration is missing: {name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint["iter"]) != iterations:
        raise ValueError(f"Checkpoint iteration {checkpoint['iter']} differs from required {iterations}")
    if stage == 1 and checkpoint.get("has_planner"):
        raise ValueError("A later-stage planner checkpoint cannot stand in for Stage 1")
    if stage >= 2 and not (checkpoint.get("has_planner") and "planner_state_dict" in checkpoint):
        raise ValueError("Planner-in-loop stage checkpoint has no trained planner")
    if stage == 3 and not (isinstance(checkpoint.get("infos"), dict) and checkpoint["infos"].get("stage") == 3):
        raise ValueError("Stage 3 checkpoint lacks infos.stage == 3 proof")
    training_state = torch.load(sidecar, map_location="cpu", weights_only=False)
    if int(training_state["iteration"]) != iterations:
        raise ValueError("Checkpoint and sidecar iterations differ")
    for key in ("wm", "critic", "opt_wm", "opt_v", "cfg"):
        if key not in training_state:
            raise ValueError(f"Training sidecar lacks {key}")
    if stage >= 2 and "planner_trainer" not in training_state:
        raise ValueError("Planner trainer state is missing from the sidecar")
    if stage == 3 and any(key not in training_state for key in ("critic_goal", "opt_vg", "stumble_head", "opt_stumble")):
        raise ValueError("Stage 3 goal critic/stumbling training state is missing")
    return {"path": str(path), "sha256": digest(path), "sidecar": str(sidecar),
            "sidecar_sha256": digest(sidecar), "iteration": iterations, "stage": stage,
            "checkpoint_stat": [path.stat().st_size, path.stat().st_mtime_ns],
            "sidecar_stat": [sidecar.stat().st_size, sidecar.stat().st_mtime_ns]}


def tracker_gate(report_path, dataset, split, checkpoint, episodes, mode=None):
    import yaml
    report = json_read(report_path)
    if mode is not None and report.get("mode") != mode:
        raise ValueError("Evaluation report mode differs from the requested gate")
    if Path(report["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("Evaluation report belongs to a different checkpoint")
    metadata_path = Path(dataset) / ("metadata.yaml" if split == "train" else "holdout/metadata.yaml")
    metadata = yaml.safe_load(metadata_path.read_text())
    if report.get("metadata_sha256") and report["metadata_sha256"] != digest(metadata_path):
        raise ValueError("Evaluation used different dataset metadata")
    expected = {Path(entry["motion_file"]).stem for entry in metadata["motion_files"]}
    rows = report["clips"]
    if {row["clip"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("Evaluation report did not cover exactly every expected clip")
    provenance = json_read(Path(dataset) / "provenance.json")
    train_hashes, holdout_hashes = (set(provenance["split"][key]) for key in ("train_source_hashes", "holdout_source_hashes"))
    if train_hashes & holdout_hashes:
        raise ValueError("Training and holdout source-motion hashes overlap")
    source_by_clip = {Path(item["motion_file"]).stem: item["source_motion_sha256"] for item in provenance["selected_windows"]}
    grouped = {}
    for row in rows:
        if int(row["episodes"]) != episodes or row["unfinished"]:
            raise ValueError(f"Incomplete fixed evaluation cohort: {row['clip']}")
        success = float(row["success"])
        if not math.isfinite(success) or not 0 <= success <= 1:
            raise ValueError("Invalid success rate")
        grouped.setdefault(source_by_clip[row["clip"]], []).append(success)
    required = train_hashes if split == "train" else holdout_hashes
    if set(grouped) != required:
        raise ValueError("Evaluation did not cover the expected original source-motion cohort")
    by_source = {key: sum(values) / len(values) for key, values in grouped.items()}
    return {"source_macro_success": sum(by_source.values()) / len(by_source), "source_count": len(by_source),
            "window_count": len(rows), "episodes_per_window": episodes, "per_source_success": by_source,
            "report": str(report_path), "report_sha256": digest(report_path), "metric": "success, not survival"}


class Pipeline:
    def __init__(self, config_path, retry=False):
        self.config_path = Path(config_path).resolve()
        self.config = json_read(self.config_path)
        self.root = Path(self.config["state_dir"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.json"
        self.config_hash = digest(self.config_path)
        self.state = json_read(self.status_path) if self.status_path.exists() else {
            "schema_version": 1, "created_at": now(), "config": str(self.config_path), "config_sha256": self.config_hash,
            "status": "ready", "steps": {}, "controller_sha256": digest(Path(__file__).resolve()),
            "hardware_control": "prohibited by this pipeline"}
        if self.state["config_sha256"] != self.config_hash:
            raise ValueError("Configuration changed: use a new state_dir so previous receipts cannot silently apply to new work")
        if self.state["status"] == "needs_attention" and retry:
            failed = self.state.get("active_step")
            if failed:
                entry = self.state["steps"].get(failed, {})
                if entry.get("exit_file") and Path(entry["exit_file"]).exists():
                    # Keep the failed evidence, and require a new step attempt.
                    entry.setdefault("previous_attempts", []).append({key: value for key, value in entry.items() if key != "previous_attempts"})
                    for key in ("exit_file", "process_file", "worker_pid", "worker_birth", "result"):
                        entry.pop(key, None)
                entry["status"] = "pending"
            self.state["status"] = "ready"
        self.python = self.config.get("python", sys.executable)
        self.repo = Path(self.config["training_repo"]).resolve()
        self.dataset = Path(self.config["dataset"]).resolve()
        self.launcher = self.repo / "tools/acp_g1/run_stages.py"

    def save(self):
        self.state["updated_at"] = now()
        json_write(self.status_path, self.state)

    def attention(self, key, message):
        self.state.update(status="needs_attention", active_step=key, reason=message)
        self.state["steps"].setdefault(key, {}).update(status="needs_attention", reason=message)
        self.save()
        return False

    def process(self, key, command, env=None, external=None):
        entry = self.state["steps"].setdefault(key, {"status": "pending"})
        self.state["active_step"] = key
        if entry["status"] == "done":
            return True
        if entry["status"] == "needs_attention":
            return False
        if "exit_file" not in entry:
            if external:
                entry.update(exit_file=str(external["exit_file"]), external=True,
                             process_file=str(external.get("process_file", "")),
                             worker_pid=external.get("pid"), worker_birth=external.get("pid_birth") or birth(external.get("pid")),
                             log=external.get("log"))
            else:
                attempt = len(entry.get("previous_attempts", [])) + 1
                prefix = self.root / f"{key}.attempt{attempt}"
                spec = {"command": command, "cwd": str(self.repo), "env": env or {}, "log": str(prefix) + ".log",
                        "exit_file": str(prefix) + ".exit.json", "process_file": str(prefix) + ".process.json"}
                spec_path = Path(str(prefix) + ".spec.json")
                json_write(spec_path, spec)
                process = subprocess.Popen([self.python, str(Path(__file__).resolve()), "--worker-spec", str(spec_path)],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                entry.update(exit_file=spec["exit_file"], process_file=spec["process_file"], log=spec["log"],
                             command=command, worker_pid=process.pid, worker_birth=birth(process.pid))
            entry["status"] = "running"
            self.save()
        receipt_path = Path(entry["exit_file"])
        if receipt_path.is_file():
            receipt = json_read(receipt_path)
            if int(receipt["returncode"]) != 0:
                return self.attention(key, f"Process exited {receipt['returncode']}; inspect {entry.get('log', receipt_path)}")
            entry.update(status="done", result=receipt, completed_at=now())
            self.save()
            return True
        live = alive(entry.get("worker_pid"), entry.get("worker_birth"))
        process_file = Path(entry["process_file"]) if entry.get("process_file") else None
        if process_file and process_file.is_file():
            child = json_read(process_file)
            live |= alive(child.get("pid"), child.get("birth"))
        if not live:
            # Allow a newly spawned worker one scheduling interval to write PID evidence.
            if not entry.get("checked_once"):
                entry["checked_once"] = True
                self.save()
                return False
            return self.attention(key, f"Process disappeared without a successful exit receipt: {receipt_path}")
        self.state["status"] = "running"
        self.save()
        return False

    def advance(self):
        if self.state["status"] == "needs_attention":
            return False
        validation = json_read(self.dataset / "contacts_validation.json")
        if not validation.get("passed") or not validation.get("source_split_disjoint"):
            return self.attention("dataset", "Dataset labels/split validation did not pass")
        if "dataset_receipt" not in self.state:
            for manifest_name in ("FILES.sha256", "CONTACTS.sha256"):
                for line in (self.dataset / manifest_name).read_text().splitlines():
                    expected, relative = line.split("  ", 1)
                    file = (self.dataset / relative).resolve()
                    if not file.is_relative_to(self.dataset) or digest(file) != expected:
                        return self.attention("dataset", f"Dataset file hash mismatch: {relative}")
            self.state["dataset_receipt"] = {name: digest(self.dataset / name) for name in
                                            ("metadata.yaml", "provenance.json", "FILES.sha256", "CONTACTS.sha256", "contacts_validation.json")}
            self.save()
        for name, expected in self.state["dataset_receipt"].items():
            if digest(self.dataset / name) != expected:
                return self.attention("dataset", f"Validated dataset manifest changed: {name}")
        through = int(self.config.get("through_stage", 2))
        checkpoints = {}
        for stage in range(1, through + 1):
            setting = self.config["stages"][str(stage)]
            run_dir = Path(setting["run_dir"]).resolve()
            iterations = int(setting.get("iterations", 30000 if stage == 1 else 10000))
            key = f"stage{stage}.train"
            command = [self.python, str(self.launcher), f"stage{stage}", "--dataset", str(self.dataset), "--run-dir", str(run_dir),
                       "--num-envs", str(setting.get("num_envs", 4096 if stage == 1 else 2048)), "--iterations", str(iterations),
                       "--gpu", str(self.config.get("train_gpu", 0))]
            if stage > 1:
                command += ["--resume-checkpoint", checkpoints[stage - 1]["path"]]
            if not self.process(key, command, external=setting.get("attach")):
                return False
            launcher_exit_path = run_dir / "exit_status.json"
            try:
                launcher_exit = json_read(launcher_exit_path)
                if int(launcher_exit["returncode"]) != 0 or int(launcher_exit["completed_iterations"]) != iterations:
                    raise ValueError("Launcher did not successfully complete the required iteration count")
                checkpoint = Path(launcher_exit["checkpoint"]).resolve()
                if checkpoint.parent != run_dir:
                    raise ValueError("Launcher checkpoint is outside its own run directory")
                previous = self.state["steps"][key].get("checkpoint")
                if previous:
                    for name, stat_name in (("path", "checkpoint_stat"), ("sidecar", "sidecar_stat")):
                        stat = Path(previous[name]).stat()
                        if [stat.st_size, stat.st_mtime_ns] != previous[stat_name]:
                            raise ValueError("Validated checkpoint/sidecar changed on disk")
                    receipt = previous
                else:
                    receipt = checkpoint_receipt(checkpoint, stage, iterations)
                self.state["steps"][key]["checkpoint"] = receipt
                checkpoints[stage] = receipt
                self.save()
            except Exception as error:
                return self.attention(key, str(error))
            if stage <= 2:
                for split in ("train", "holdout"):
                    for mode in ("paper", "robust"):
                        eval_key = f"stage{stage}.eval.{split}.{mode}"
                        output = self.root / f"{eval_key}.json"
                        dataset = self.dataset if split == "train" else self.dataset / "holdout"
                        episodes = int(self.config.get("evaluation_episodes", 8))
                        command = [self.python, "-m", "instinct_mj.scripts.acp.eval_tracker", str(checkpoint),
                                   "--task", TASKS[stage], "--mode", mode, "--num-envs", str(self.config.get("evaluation_envs", 128)),
                                   "--episodes", str(episodes), "--max-steps", "1500", "--out", str(output)]
                        env = {"ACP_DATASET": str(dataset), "CUDA_VISIBLE_DEVICES": str(self.config.get("eval_gpu", 1)), "MUJOCO_GL": "egl"}
                        if not self.process(eval_key, command, env):
                            return False
                        try:
                            result = tracker_gate(output, self.dataset, split, checkpoint, episodes, mode)
                            threshold = float(self.config.get("thresholds", {}).get(f"{split}.{mode}", THRESHOLDS[f"{split}.{mode}"]))
                            result.update(threshold=threshold, passed=result["source_macro_success"] >= threshold)
                            self.state["steps"][eval_key]["gate"] = result
                            self.save()
                            if not result["passed"]:
                                return self.attention(eval_key, f"Source-macro success {result['source_macro_success']:.4f} < {threshold:.4f}; later stages were not launched")
                        except Exception as error:
                            return self.attention(eval_key, str(error))
            else:
                output = self.root / "stage3.goal.json"
                command = [self.python, str(self.repo / "tools/acp_g1/eval_goal.py"), "--checkpoint", str(checkpoint),
                           "--dataset", str(self.dataset), "--output", str(output), "--episodes", str(self.config.get("goal_episodes", 128)),
                           "--gpu", str(self.config.get("eval_gpu", 1))]
                if not self.process("stage3.goal", command, {"CUDA_VISIBLE_DEVICES": str(self.config.get("eval_gpu", 1)), "MUJOCO_GL": "egl"}):
                    return False
                result = json_read(output)
                rate = float(result["overall"]["success"])
                count = int(result.get("episode_count", result["overall"].get("episode_count", 0)))
                if not math.isfinite(rate) or rate < float(self.config.get("goal_threshold", .9)) or count != int(self.config.get("goal_episodes", 128)):
                    return self.attention("stage3.goal", "Native flat goal-reaching-and-stop evaluation did not meet its success/cohort gate")
                bundle = Path(self.config["bundle_dir"]).resolve()
                command = [self.python, self.config["exporter"], "--instinct-mj", str(self.repo), "--dataset", str(self.dataset),
                           "--task", "Instinct-ACP-Stage3-Flat-G1-Play-v1", "--checkpoint", str(checkpoint),
                           "--fixed-joints", self.config["fixed_joints"], "--output", str(bundle)]
                if not self.process("stage3.export", command, {"CUDA_VISIBLE_DEVICES": str(self.config.get("eval_gpu", 1)), "MUJOCO_GL": "egl"}):
                    return False
                manifest = json_read(bundle / "manifest.json")
                if manifest.get("stage") != 3 or not manifest.get("parity", {}).get("passed") or digest(bundle / "policy.onnx") != manifest.get("policy_sha256"):
                    return self.attention("stage3.export", "Export manifest, native/ONNX parity, or policy hash failed")
                checks = self.config.get("post_export_checks", [])
                if {check["name"] for check in checks} != {"replay", "deployment_mujoco"}:
                    return self.attention("post_export", "Configure both replay and deployment_mujoco check commands before declaring simulation validation")
                for check in checks:
                    command = [value.format(bundle=str(bundle), checkpoint=str(checkpoint), dataset=str(self.dataset)) for value in check["command"]]
                    if not self.process("post_export." + check["name"], command, check.get("env")):
                        return False
                    if not json_read(check["report"])["passed"]:
                        return self.attention("post_export." + check["name"], "Deployment validation report did not pass")
        self.state.update(status=f"stage{through}_simulation_validated", active_step=None,
                          reason="Training/evaluation gates passed. This controller does not authorize or start hardware control.")
        self.save()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--retry-needs-attention", action="store_true", help="Retry failed step while preserving its previous evidence")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec:
        raise SystemExit(worker(args.worker_spec))
    if not args.config:
        parser.error("--config is required")
    if not args.execute:
        config = json_read(args.config)
        print(json.dumps({"mode": "plan_only", "config": config, "gates": THRESHOLDS,
                          "sequence": ["Stage 1", "train/holdout paper+robust source-macro success", "Stage 2 planner in loop",
                                       "same four gates", "Stage 3 Flat", "native goal+stop gate", "ONNX parity export", "replay", "deployment MuJoCo"],
                          "hardware_control": "never"}, indent=2))
        return
    pipeline = Pipeline(args.config, args.retry_needs_attention)
    with (pipeline.root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                completed = pipeline.advance()
            except Exception as error:
                pipeline.attention(pipeline.state.get("active_step") or "controller", str(error))
                completed = False
            print(json.dumps({key: pipeline.state.get(key) for key in ("status", "active_step", "reason", "updated_at")}), flush=True)
            if completed or pipeline.state["status"] == "needs_attention" or args.once:
                break
            time.sleep(max(1, min(args.poll_seconds, 60)))
    if pipeline.state["status"] == "needs_attention":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
