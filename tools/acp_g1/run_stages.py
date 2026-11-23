#!/usr/bin/env python3
"""Reproducible ACP training, smoke, preflight and same-stage resume.

Each stage requires an explicit invocation; this launcher never advances stages.
Use this file with the worktree's .venv/bin/python.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
TASKS = {
    "smoke": "Instinct-ACP-Track-G1-v1",
    "preflight": "Instinct-ACP-Track-G1-v1",
    "stage1": "Instinct-ACP-Track-G1-v1",
    "stage2": "Instinct-ACP-Stage2-G1-v1",
    "stage3": "Instinct-ACP-Stage3-Flat-G1-v1",
}
_RUN_DIR: Path | None = None
_TARGET_ITERATIONS = 0
_RESUME_CHECKPOINT: Path | None = None

# Keep the chosen stage's handover schedules and counters. The saved policy,
# normalizers, world model, critic and optimizer-related settings are inherited.
STAGE_FIELDS = {
    "contact_source", "train_planner", "restart_iteration", "wm_only_iters",
    "eta_obs_start_iter", "eta_rew_start_iter", "eta_ann_iters",
    "planner_updates_per_iter", "lazy_exact_heightmap", "experiment_name",
    "run_name", "max_iterations", "save_interval", "log_interval", "load_run", "freeze_tracker", "planner_seed_initial_support",
    "load_checkpoint", "resume", "device", "profile_phases",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def verify_frozen_tracker(before: dict, after: dict) -> int:
    """Reject any change to actor or policy normalization, including counts."""
    import torch
    checked = 0
    for group in ("model_state_dict", "policy_normalizer_state_dict"):
        a, b = before[group], after[group]
        if a.keys() != b.keys():
            raise ValueError(f"Frozen tracker keys changed: {group}")
        for name in a:
            if a[name].dtype != b[name].dtype or not torch.equal(a[name], b[name]):
                raise ValueError(f"Frozen tracker tensor changed: {group}.{name}")
            checked += 1
    return checked


def inherit_agent_config(target, saved: dict) -> None:
    """Restore compatible dataclass fields without replacing nested cfg objects."""
    for key, value in saved.items():
        if key in STAGE_FIELDS or not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            inherit_agent_config(current, value)
        elif isinstance(current, dict) and isinstance(value, dict):
            for name, item in value.items():
                if name in current and is_dataclass(current[name]) and isinstance(item, dict):
                    inherit_agent_config(current[name], item)
                else:
                    current[name] = deepcopy(item)
        else:
            setattr(target, key, deepcopy(value))


def override_actor_lr_scale(agent, value: float | None) -> None:
    """Change only the actor multiplier after restoring the saved config."""
    if value is not None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("actor-lr-scale must be finite and positive")
        agent.pi_lr_scale = value


def override_policy_batch(agent, value: int | None) -> None:
    """Limit imagined starts after resume without changing WM/planner batches."""
    if value is not None:
        if value <= 0:
            raise ValueError("pi-batch must be positive")
        agent.pi_batch = value


def override_reward_handover_start(agent, value: int | None) -> None:
    """Delay reward takeover independently of the observation schedule."""
    if value is not None:
        if type(value) is not int or value < 0:
            raise ValueError("reward-handover-start must be a nonnegative integer")
        if agent.contact_source == "mixed" and value <= agent.eta_obs_start_iter:
            raise ValueError("reward handover must start after observation handover")
        agent.eta_rew_start_iter = value


def override_planner_experiment(agent, planner_lr=None, hold_at=None, release_hold=False, freeze=None):
    """Explicit Stage 2 controls, applied after the checkpoint config is restored."""
    if planner_lr is not None:
        if not math.isfinite(planner_lr) or planner_lr <= 0:
            raise ValueError("planner-lr must be finite and positive")
        agent.planner_lr = planner_lr
    if hold_at is not None:
        if not isinstance(hold_at, int) or hold_at < 0 or release_hold:
            raise ValueError("handover hold must be a nonnegative iteration, exclusive with release")
        agent.handover_hold_iteration = hold_at
    if release_hold:
        agent.handover_hold_iteration = None
    if freeze is not None:
        agent.freeze_planner = freeze


def verify_frozen_planner(before: dict, after: dict) -> int:
    import torch
    a, b = before["planner_state_dict"], after["planner_state_dict"]
    if a.keys() != b.keys():
        raise ValueError("Frozen planner keys changed")
    for name, value in a.items():
        if value.dtype != b[name].dtype or not torch.equal(value, b[name]):
            raise ValueError(f"Frozen planner tensor changed: {name}")
    return len(a)


def exit_status(returncode: int, error: str | None = None) -> dict:
    """Publish only checkpoint/sidecar pairs that actually exist."""
    candidates = []
    if _RUN_DIR is not None:
        for path in _RUN_DIR.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", path.name)
            if match and path.with_name(f"acp_state_{match[1]}.pt").is_file():
                candidates.append((int(match[1]), path.resolve()))
    if not candidates and _RESUME_CHECKPOINT is not None:
        match = re.fullmatch(r"model_(\d+)\.pt", _RESUME_CHECKPOINT.name)
        if match:
            candidates.append((int(match[1]), _RESUME_CHECKPOINT))
    completed, checkpoint = max(candidates, default=(0, None))
    if returncode == 0 and completed < _TARGET_ITERATIONS:
        returncode, error = 130, "training returned before its target iteration"
    status = dict(returncode=returncode, completed_iterations=completed,
                  target_iterations=_TARGET_ITERATIONS,
                  checkpoint=str(checkpoint) if checkpoint else None,
                  error=error, finished_utc=datetime.now(timezone.utc).isoformat())
    if _RUN_DIR is not None:
        _RUN_DIR.mkdir(parents=True, exist_ok=True)
        temporary = _RUN_DIR / ".exit_status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(_RUN_DIR / "exit_status.json")
    return status


def main() -> int:
    global _RUN_DIR, _TARGET_ITERATIONS, _RESUME_CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(TASKS))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--planner-refit", action="store_true",
                        help="Stage 2 only: frozen actor/normalizer, reference-contact collection, planner learning")
    parser.add_argument("--seed-initial-support", action="store_true",
                        help="Reference episodes: exclude initial support settling from new-contact labels")
    parser.add_argument("--actor-lr-scale", type=float,
                        help="Override only pi_lr_scale after saved configuration is restored (not the WM/critic/planner LR)")
    parser.add_argument("--pi-batch", type=int,
                        help="Policy imagination batch after resume; retain the saved value unless explicitly set")
    parser.add_argument("--planner-lr", type=float, help="Stage 2 planner Adam LR after optimizer resume; actor/WM/critic LRs unchanged")
    parser.add_argument("--reward-handover-start", type=int,
                        help="Stage 2: override only the reward eta start after resume; observation eta keeps its stage schedule")
    handover = parser.add_mutually_exclusive_group()
    handover.add_argument("--handover-hold-iteration", type=int, help="Stage 2: cap the eta schedule clock, preserving actual iteration counters")
    handover.add_argument("--release-handover-hold", action="store_true", help="Stage 2: resume normal eta schedule progression")
    parser.add_argument("--freeze-planner", action=argparse.BooleanOptionalAction, default=None,
                        help="Stage 2: stop planner optimizer updates while continuing tracker training")
    parser.add_argument("--release-cuda-cache", action=argparse.BooleanOptionalAction, default=None,
                        help="Return unused Torch cache to MuJoCo/Warp between iterations; inherit saved setting when omitted")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--heightmap-cache-dir", type=Path,
                        default=Path(os.environ.get("ACP_HEIGHTMAP_CACHE_DIR", "~/.cache/instinct_mj/supermimic_heightmaps")),
                        help="Host-local cache; overrides cache paths inherited from another training host")
    args = parser.parse_args()
    dataset, run_dir = args.dataset.expanduser().resolve(), args.run_dir.expanduser().resolve()
    task = TASKS[args.mode]
    if not (dataset / "metadata.yaml").is_file():
        parser.error(f"missing {dataset / 'metadata.yaml'}")
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        parser.error(f"run directory must be empty: {run_dir}")
    _RUN_DIR = run_dir
    if args.mode in ("stage2", "stage3") and args.resume_checkpoint is None:
        parser.error("Stage 2/3 require a previous-stage checkpoint and ACP sidecar")
    if args.planner_refit and args.mode != "stage2":
        parser.error("--planner-refit requires stage2 and its checkpoint/sidecar")
    if any((args.planner_lr is not None, args.handover_hold_iteration is not None,
            args.release_handover_hold, args.freeze_planner is not None,
            args.reward_handover_start is not None)) and args.mode != "stage2":
        parser.error("planner experiment controls require stage2")
    if args.planner_refit and args.freeze_planner:
        parser.error("planner-refit and freeze-planner cannot be combined")
    if args.seed_initial_support and args.mode not in ("stage2", "stage3"):
        parser.error("--seed-initial-support requires stage2 or stage3")
    if args.num_envs is not None and args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.iterations is not None and args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.actor_lr_scale is not None and (not math.isfinite(args.actor_lr_scale) or args.actor_lr_scale <= 0):
        parser.error("--actor-lr-scale must be finite and positive")
    if args.pi_batch is not None and args.pi_batch <= 0:
        parser.error("--pi-batch must be positive")
    if args.planner_lr is not None and (not math.isfinite(args.planner_lr) or args.planner_lr <= 0):
        parser.error("--planner-lr must be finite and positive")
    if args.handover_hold_iteration is not None and args.handover_hold_iteration < 0:
        parser.error("--handover-hold-iteration must be nonnegative")
    if args.reward_handover_start is not None and args.reward_handover_start < 0:
        parser.error("--reward-handover-start must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["ACP_DATASET"] = str(dataset)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("WARP_CACHE_PATH", "/root/autodl-tmp/acp-g1/cache/warp")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/root/autodl-tmp/acp-g1/cache/torchinductor")
    os.environ.setdefault("MPLCONFIGDIR", "/root/autodl-tmp/acp-g1/cache/matplotlib")

    import yaml
    meta = yaml.safe_load((dataset / "metadata.yaml").read_text())
    entries = meta.get("motion_files", [])
    if not entries:
        parser.error("dataset contains no motion files")
    for entry in entries:
        motion = dataset / entry["motion_file"]
        sidecar = motion.with_suffix(".contacts.npz")
        if not motion.is_file() or not sidecar.is_file():
            parser.error(f"incomplete motion/contact pair: {motion}")
    for terrain in meta.get("terrains", []):
        if terrain.get("terrain_file") and not (dataset / terrain["terrain_file"]).is_file():
            parser.error(f"missing terrain: {terrain['terrain_file']}")

    # Match the repository's train_on_dataset process-local override.
    import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as acp_g1
    acp_g1.MOTION_FOLDER = str(dataset)
    buffer = acp_g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
    buffer.path, buffer.metadata_yaml = str(dataset), str(dataset / "metadata.yaml")
    from instinct_mj.scripts.instinct_rl.train import TrainConfig, run_train
    cfg = TrainConfig.from_task(task)
    inherited_agent = None
    if args.resume_checkpoint:
        checkpoint = args.resume_checkpoint.expanduser().resolve()
        if (not re.fullmatch(r"model_\d+\.pt", checkpoint.name)
                or not checkpoint.is_file()
                or not checkpoint.with_name(checkpoint.name.replace("model_", "acp_state_", 1)).is_file()):
            parser.error("resume requires a model checkpoint and its matching ACP sidecar")
        inherited_agent = checkpoint.parent / "params" / "agent.yaml"
        if not inherited_agent.is_file():
            parser.error(f"missing saved agent configuration: {inherited_agent}")
        import torch
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if args.mode == "stage3" and not saved.get("has_planner", False):
            parser.error("Stage 3 requires a checkpoint containing a trained planner")
        if args.mode == "stage2" and (saved.get("infos") or {}).get("stage") == 3:
            parser.error("a Stage 3 checkpoint cannot initialize Stage 2")
        if args.planner_refit and not saved.get("has_planner", False):
            parser.error("--planner-refit requires an existing Stage 2 planner")
        if args.freeze_planner and not saved.get("has_planner", False):
            parser.error("--freeze-planner requires an existing trained planner")
        if args.mode in ("stage1", "smoke", "preflight") and saved.get("has_planner", False):
            parser.error("Stage 1 cannot resume a checkpoint from a later stage")
        inherit_agent_config(cfg.agent, yaml.safe_load(inherited_agent.read_text()))
        cfg.agent.load_run = str(checkpoint.parent)
        cfg.agent.load_checkpoint = checkpoint.name
        cfg.agent.resume = True
        _RESUME_CHECKPOINT = checkpoint
    cfg.agent.freeze_tracker = args.planner_refit
    override_actor_lr_scale(cfg.agent, args.actor_lr_scale)
    if args.mode == "stage3":
        cfg.agent.handover_hold_iteration = None
        cfg.agent.freeze_planner = False
    override_planner_experiment(cfg.agent, args.planner_lr, args.handover_hold_iteration,
                                args.release_handover_hold, args.freeze_planner)
    override_reward_handover_start(cfg.agent, args.reward_handover_start)
    if args.planner_refit and cfg.agent.freeze_planner:
        parser.error("planner-refit cannot use a frozen planner; pass --no-freeze-planner to release it")
    cfg.agent.planner_seed_initial_support = args.seed_initial_support
    if args.planner_refit:
        cfg.agent.contact_source = "reference"
        cfg.agent.train_planner = True
        cfg.agent.cuda_graph_policy = False
        cfg.agent.compile_policy = False
    cfg.agent.heightmap_cache_dir = str(args.heightmap_cache_dir.expanduser().resolve())
    cfg.agent.device = "cuda:0"
    diagnostic = args.mode in ("smoke", "preflight")
    cfg.agent.log_interval = 1 if diagnostic else 10
    cfg.agent.save_interval = 1 if diagnostic else 1000
    cfg.env.scene.num_envs = args.num_envs or (32 if args.mode == "smoke" else cfg.env.scene.num_envs)
    cfg.agent.max_iterations = args.iterations or (6 if args.mode == "preflight" else 2 if args.mode == "smoke" else cfg.agent.max_iterations)
    _TARGET_ITERATIONS = cfg.agent.max_iterations
    cfg.agent.profile_phases = diagnostic
    if args.mode == "smoke":
        for key, value in dict(buffer_steps=64, warmup_steps=16, wm_batch=32,
                               wm_batches=1, pi_batch=32, horizon=8,
                               wm_only_iters=0, compile=False, compile_policy=False,
                               cuda_graph_wm=False, cuda_graph_policy=False).items():
            setattr(cfg.agent, key, value)
    elif args.mode == "preflight":
        cfg.agent.wm_only_iters = 0  # Exercise the full actor update before long training.
    override_policy_batch(cfg.agent, args.pi_batch)
    if args.release_cuda_cache is not None:
        cfg.agent.release_cuda_cache_each_iteration = args.release_cuda_cache

    run_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in dataset.rglob("*") if p.is_file() and p.suffix.lower() in (".npz", ".yaml", ".stl", ".obj", ".json")]
    dataset_files = [{"path": str(p.relative_to(dataset)), "size": p.stat().st_size,
                      "sha256": sha256(p)} for p in sorted(files)]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                    task=task, mode=args.mode, python=sys.executable,
                    source_head=git("rev-parse", "HEAD"), source_status=git("status", "--short"),
                    source_diff_sha256=hashlib.sha256(git("diff", "HEAD", "--", "src").encode()).hexdigest(),
                    lock_sha256=sha256(REPO / "uv.lock"), launcher_sha256=sha256(Path(__file__)),
                    dataset=str(dataset), motion_count=len(entries), dataset_files=dataset_files,
                    gpu_physical_index=args.gpu, num_envs=cfg.env.scene.num_envs,
                    total_iterations=cfg.agent.max_iterations,
                    inherited_agent_config=str(inherited_agent) if inherited_agent else None,
                    inherited_agent_sha256=sha256(inherited_agent) if inherited_agent else None,
                    resume_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None,
                    planner_refit=args.planner_refit,
                    seed_initial_support=args.seed_initial_support,
                    actor_lr_scale=cfg.agent.pi_lr_scale,
                    planner_lr=cfg.agent.planner_lr,
                    eta_obs_start_iter=cfg.agent.eta_obs_start_iter,
                    eta_rew_start_iter=cfg.agent.eta_rew_start_iter,
                    eta_ann_iters=cfg.agent.eta_ann_iters,
                    handover_hold_iteration=cfg.agent.handover_hold_iteration,
                    freeze_planner=cfg.agent.freeze_planner,
                    policy_batch_size=cfg.agent.pi_batch,
                    world_model_batch_size=cfg.agent.wm_batch,
                    planner_batch_size=cfg.agent.planner_batch,
                    release_cuda_cache_each_iteration=cfg.agent.release_cuda_cache_each_iteration,
                    heightmap_cache_dir=cfg.agent.heightmap_cache_dir,
                    effective_actor_lr=cfg.agent.lr * cfg.agent.pi_lr_scale,
                    collection_contact_source=cfg.agent.contact_source)
    (run_dir / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "dataset_files"}, indent=2), flush=True)
    os.chdir(REPO)
    started = time.monotonic()
    run_train(task, cfg, run_dir)
    import torch
    freeze_check = None
    planner_freeze_check = None
    if args.planner_refit:
        final_checkpoint = run_dir / f"model_{cfg.agent.max_iterations}.pt"
        after = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
        freeze_check = dict(tensors_checked=verify_frozen_tracker(saved, after),
                            source_checkpoint=str(checkpoint), source_sha256=sha256(checkpoint),
                            final_checkpoint=str(final_checkpoint), final_sha256=sha256(final_checkpoint),
                            actor_and_policy_normalizer_bitwise_unchanged=True)
        (run_dir / "tracker_freeze_check.json").write_text(json.dumps(freeze_check, indent=2) + "\n")
    if cfg.agent.freeze_planner:
        final_checkpoint = run_dir / f"model_{cfg.agent.max_iterations}.pt"
        after = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
        planner_freeze_check = dict(tensors_checked=verify_frozen_planner(saved, after),
                                   source_sha256=sha256(checkpoint), final_sha256=sha256(final_checkpoint),
                                   planner_bitwise_unchanged=True)
        (run_dir / "planner_freeze_check.json").write_text(json.dumps(planner_freeze_check, indent=2) + "\n")
    result = dict(elapsed_seconds=time.monotonic() - started,
                  max_memory_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  max_memory_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  gpu_total_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                  checkpoints=sorted(p.name for p in run_dir.glob("model_*.pt")),
                  sidecars=sorted(p.name for p in run_dir.glob("acp_state_*.pt")),
                  frozen_tracker_verified=freeze_check is not None,
                  frozen_planner_verified=planner_freeze_check is not None)
    (run_dir / "run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return exit_status(0)["returncode"]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            code = 130
        elif isinstance(exc, SystemExit):
            code = exc.code if isinstance(exc.code, int) else 1
        else:
            code = 1
        if code and (_RUN_DIR is None or not (_RUN_DIR / "exit_status.json").is_file()):
            exit_status(code, f"{type(exc).__name__}: {exc}")
        raise
