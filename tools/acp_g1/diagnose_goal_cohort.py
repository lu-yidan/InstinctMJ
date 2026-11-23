#!/usr/bin/env python3
"""Paired, zero-update flat-goal diagnosis; deliberately separate from acceptance.

A real Stage 2 checkpoint is allowed only as an explicitly named baseline. The
formal eval_goal.py Stage 3 guard and pipeline thresholds are not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from diagnose_abc import array_digest, tensor_storage
from run_stages import inherit_agent_config

CASES = ("forward", "turn", "stop")


def fixed_targets(spawn, yaw, ground_z):
    """World targets: 1/1.5/2 m forward or +/-90 degree turn; stationary stop."""
    n = len(spawn)
    case = np.arange(n) % 3
    distance = np.asarray([1., 1.5, 2.])[np.arange(n) // 3 % 3]
    bearing = yaw + np.where(case == 1, np.where(np.arange(n) // 3 % 2, -1., 1.) * math.pi / 2, 0.)
    distance = np.where(case == 2, 0., distance)
    target = np.array(spawn, copy=True)
    target[:, :2] += np.stack((np.cos(bearing), np.sin(bearing)), -1) * distance[:, None]
    target[:, 2] = ground_z + .78
    heading = np.arctan2(np.sin(bearing), np.cos(bearing))
    return target, heading, case


def hold_metrics(standing, alive, hold_frames):
    """A hold at any time and a hold through the final frame are distinct."""
    held = np.zeros(standing.shape[1], dtype=np.int64)
    ever = np.zeros_like(held, dtype=bool)
    for good, live in zip(standing, alive):
        held = np.where(good & live, held + 1, 0)
        ever |= held >= hold_frames
    return ever, held >= hold_frames


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--role", choices=("stage2-baseline", "stage3"), required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--initial-only", action="store_true", help="Verify cohort restoration without advancing simulation")
    args = p.parse_args()
    if args.episodes < 3:
        p.error("At least three episodes are needed to cover all goal types")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("Output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    dataset = args.dataset.expanduser().resolve(strict=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import yaml
    import instinct_mj.tasks  # noqa: F401
    from instinct_mj.tasks.acp.config.g1 import acp_tracking_cfg
    from instinct_mj.scripts.instinct_rl.play import _disable_headless_debug_visualization
    from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls

    # Explicit full IEEE FP32, including child conv precision (PyTorch 2.10).
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stage3 = isinstance(ck.get("infos"), dict) and ck["infos"].get("stage") == 3
    if not ck.get("has_planner") or stage3 != (args.role == "stage3"):
        raise ValueError("Checkpoint stage does not match the explicit diagnostic role")
    torch.manual_seed(args.seed)
    task = "Instinct-ACP-Stage3-Flat-G1-Play-v1"
    acp_tracking_cfg.MOTION_FOLDER = str(dataset)
    buffer = acp_tracking_cfg.motion_reference_cfg.motion_buffers["TerrainMotion"]
    buffer.path, buffer.metadata_yaml = str(dataset), str(dataset / "metadata.yaml")
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = args.episodes, args.seed, False
    cfg.curriculum.pop("goal_terrain", None)
    _disable_headless_debug_visualization(cfg)
    if cfg.scene.arena.types != ("flat",):
        raise ValueError("Diagnostic requires the dedicated flat arena")
    agent = load_instinct_rl_cfg(task)
    inherit_agent_config(agent, yaml.safe_load((checkpoint.parent / "params/agent.yaml").read_text()))
    agent.contact_source, agent.train_planner = "planner", True
    agent.freeze_tracker, agent.freeze_planner, agent.handover_hold_iteration = False, False, None
    agent.compile, agent.compile_policy = False, False
    raw = load_env_cls(task)(cfg=cfg, device="cuda:0")
    vec = load_vecenv_cls(task)(raw, policy_group=agent.policy_observation_group, critic_group=agent.critic_observation_group)
    cpu = lambda x: tensor_storage(x).detach().cpu().numpy().copy()
    dev = "cuda:0"
    ids = torch.arange(args.episodes, device=dev)
    started = time.monotonic()
    try:
        runner = load_runner_cls(task)(vec, agent.to_dict(), log_dir=None, device=dev)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=dev)
        graph = runner.policy_graph(False).eval()
        goal = raw.command_manager.get_term("acp_goal")
        ref = raw.command_manager.get_term("acp_reference_contacts")
        if not goal.is_goal.all():
            raise ValueError("Play cohort must contain only goal environments")
        goal.level[:] = ids % 10
        obs, _ = vec.reset()
        state_fields = {}
        for name in ("qpos", "qvel", "qacc_warmstart", "act", "ctrl", "qfrc_applied", "xfrc_applied", "time"):
            value = tensor_storage(getattr(raw.sim.data, name, None))
            if isinstance(value, torch.Tensor):
                state_fields["physics." + name] = value
        for name in ("_action", "_prev_action", "_prev_prev_action"):
            state_fields["action." + name] = getattr(raw.action_manager, name)
        for name in ("_raw_actions", "_processed_actions"):
            state_fields["joint_action." + name] = getattr(raw.action_manager.get_term("joint_pos"), name)
        for name in ("joint_pos_target", "joint_vel_target", "joint_effort_target"):
            state_fields["robot." + name] = getattr(raw.scene["robot"].data, name)

        created = not args.cohort.exists()
        if created:
            # Center the starting pelvis on its own flat tile. All targets stay
            # within 2 m of the center, away from the tile's boundary.
            qpos = state_fields["physics.qpos"]
            center = goal.tables.origin[goal.tile_id]
            qpos[:, :2] = center[:, :2]
            state_fields["physics.qvel"].zero_()
            q = cpu(qpos[:, 3:7])
            yaw = np.arctan2(2 * (q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
            targets, headings, cases = fixed_targets(cpu(qpos[:, :3]), yaw, cpu(center[:, 2]))
            state = {key:cpu(value) for key,value in state_fields.items()}
            state.update(targets=targets, headings=headings, cases=cases, levels=cpu(goal.level))
        else:
            with np.load(args.cohort, allow_pickle=False) as archive:
                state = {key:archive[key].copy() for key in archive.files}
            if state["targets"].shape != (args.episodes, 3):
                raise ValueError("Cohort size differs")
        for key, value in state_fields.items():
            value.copy_(torch.as_tensor(state[key], device=dev))
        goal.level.copy_(torch.as_tensor(state["levels"], device=dev))
        goal.target_pos_w.copy_(torch.as_tensor(state["targets"], device=dev))
        goal.target_yaw.copy_(torch.as_tensor(state["headings"], device=dev))
        goal.target_time.zero_()
        raw.sim.forward()
        raw.command_manager.compute(dt=0., env_ids=ids)
        for sensor in ("height_scanner", "critic_height_scanner"):
            raw.scene[sensor].reset(ids)
        raw.sim.sense()
        for key in ("physics.qacc_warmstart", "physics.act", "physics.ctrl"):
            if key in state_fields:
                state_fields[key].copy_(torch.as_tensor(state[key], device=dev))
        raw.observation_manager.reset(env_ids=ids)
        raw.obs_buf = raw.observation_manager.compute(update_history=True, env_ids=ids)
        obs, extras = vec.get_observations()
        actual = {key:cpu(value) for key,value in state_fields.items()}
        actual.update(policy_observation=cpu(obs), planner_observation=cpu(extras["observations"]["planner"]),
                      targets=cpu(goal.target_pos_w), headings=cpu(goal.target_yaw), levels=cpu(goal.level))
        # Values were saved as the exact live float32 geometry, never rounded.
        for key in ("targets", "headings"):
            state[key] = actual[key]
        actual_hash = array_digest(actual)
        if created:
            state.update(initial_sha256=np.asarray(actual_hash), seed=np.asarray(args.seed))
            args.cohort.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.cohort, **state)
        elif str(state["initial_sha256"].item()) != actual_hash:
            raise ValueError("Initial physics, history, observations or targets are not paired")
        write_json(args.output_dir / "initial.json", {"initial_state_sha256":actual_hash,
            "cohort_sha256":hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
            "source_iteration":ck["iter"], "loaded_clock":runner.it, "role":args.role,
            "agent":agent.to_dict(), "training_updates":0})
        if args.initial_only:
            print(json.dumps({"initial_state_sha256":actual_hash, "cohort_restored":not created,
                              "simulation_steps":0,"training_updates":0}),flush=True)
            return

        n, steps = args.episodes, math.ceil(15. / raw.step_dt)
        active = torch.ones(n, dtype=torch.bool, device=dev)
        fell = torch.zeros_like(active)
        numerical = torch.zeros_like(active)
        targets = goal.target_pos_w.clone()
        headings = goal.target_yaw.clone()
        trace = {key:[] for key in ("qpos", "qvel", "actions", "contacts", "occupancy", "distance", "heading_error", "speed_xy", "height_error", "standing", "alive", "arrived", "finite")}
        failure_tick = np.full(n, -1, np.int64)
        parity_error = 0.
        with torch.inference_mode():
            for tick in range(steps):
                planner_obs = torch.cat((obs[:, :375], vec.get_observations()[1]["observations"]["planner"], obs[:, 465:468]), -1)
                actions, contacts, occupancy = graph(planner_obs)
                if tick == 0:
                    parity_error = float((actions - policy(obs)).abs().max())
                    if parity_error > 1e-4:
                        raise ValueError("Instrumented graph differs from native policy: " + str(parity_error))
                finite_action = torch.isfinite(actions).all(-1)
                safe_actions = torch.where((active & finite_action)[:,None], actions, torch.zeros_like(actions))
                obs, _, done, _ = vec.step(safe_actions)
                pos, velocity, quat = ref.base_pos_w, ref.base_lin_vel_w, ref.base_quat_w
                yaw = torch.atan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]), 1-2*(quat[:,2]**2+quat[:,3]**2))
                distance = (pos[:,:2]-targets[:,:2]).norm(dim=-1)
                heading = torch.atan2(torch.sin(headings-yaw),torch.cos(headings-yaw)).abs()
                speed = velocity[:,:2].norm(dim=-1)
                height = (pos[:,2]-targets[:,2]).abs()
                finite = torch.stack([torch.isfinite(v).flatten(1).all(-1) for v in (pos, velocity, quat, obs)]).all(0) & finite_action
                upright = 1-2*(quat[:,1]**2+quat[:,2]**2) > math.cos(.7)
                new_fall = active & raw.reset_terminated & finite
                new_numerical = active & ~finite
                failed = new_fall | new_numerical
                failure_tick[cpu(failed)] = tick
                fell |= new_fall
                numerical |= new_numerical
                active &= ~failed
                arrived = (distance < .2) & (height < .1) & upright & finite & active
                standing = arrived & (heading < .25) & (speed < .15)
                values = dict(qpos=raw.sim.data.qpos, qvel=raw.sim.data.qvel, actions=actions, contacts=contacts,
                              occupancy=occupancy, distance=distance, heading_error=heading, speed_xy=speed,
                              height_error=height, standing=standing, alive=active, arrived=arrived, finite=finite)
                for key,value in values.items():
                    trace[key].append(cpu(value))
                # A terminated world is never counted again. Zero its policy
                # observation so a bad world cannot poison another batch lane.
                obs = torch.where(active[:, None], obs, torch.zeros_like(obs))
        trace = {key:np.stack(value) for key,value in trace.items()}
        np.savez_compressed(args.output_dir / "trajectories.npz", **trace, failure_tick=failure_tick, cases=state["cases"])
        ever, final = hold_metrics(trace["standing"], trace["alive"], math.ceil(1. / raw.step_dt))
        arrival = trace["arrived"].any(0)
        fallen, invalid = cpu(fell), cpu(numerical)
        rows = []
        for index in range(n):
            last = failure_tick[index] if failure_tick[index] >= 0 else steps-1
            metric = lambda key: float(trace[key][last,index]) if np.isfinite(trace[key][last,index]) else None
            rows.append(dict(episode=index, case=CASES[int(state["cases"][index])], arrival=bool(arrival[index]),
                hold_any_1s=bool(ever[index]), hold_final_1s=bool(final[index]), fell=bool(fallen[index]),
                numerical_failure=bool(invalid[index]), first_failure_tick=int(failure_tick[index]),
                endpoint_distance=metric("distance"), endpoint_heading_error=metric("heading_error"),
                endpoint_speed_xy=metric("speed_xy"), target=state["targets"][index].tolist(),
                target_heading=float(state["headings"][index])))
        def aggregate(selected):
            finite_rows = [r for r in selected if not r["numerical_failure"]]
            return dict(episodes=len(selected), arrival_rate=sum(r["arrival"] for r in selected)/len(selected),
                hold_any_1s_rate=sum(r["hold_any_1s"] for r in selected)/len(selected),
                hold_final_1s_rate=sum(r["hold_final_1s"] for r in selected)/len(selected),
                falls=sum(r["fell"] for r in selected), numerical_failures=sum(r["numerical_failure"] for r in selected),
                endpoint_distance_median=float(np.median([r["endpoint_distance"] for r in finite_rows])) if finite_rows else None,
                endpoint_heading_error_median=float(np.median([r["endpoint_heading_error"] for r in finite_rows])) if finite_rows else None,
                endpoint_speed_xy_median=float(np.median([r["endpoint_speed_xy"] for r in finite_rows])) if finite_rows else None)
        result = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            role=args.role, source_iteration=ck["iter"], task=task, seed=args.seed, training_updates=0,
            initial_state_sha256=actual_hash, cohort_sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
            inference_precision="IEEE FP32 including cuDNN conv", native_graph_action_max_abs_error=parity_error,
            overall=aggregate(rows), by_case={case:aggregate([r for r in rows if r["case"] == case]) for case in CASES},
            criteria=dict(distance=.2, heading=.25, speed=.15, height=.1, tilt=.7, hold_seconds=1., horizon_seconds=15.,
                semantics="any one-second hold and final one-second hold separately; diagnostic, not formal or deployment acceptance"),
            wall_seconds=time.monotonic()-started, episodes=rows)
        write_json(args.output_dir / "result.json", result)
        print(json.dumps({"result":str(args.output_dir / "result.json"), "overall":result["overall"]}), flush=True)
    finally:
        vec.close()


if __name__ == "__main__":
    main()
