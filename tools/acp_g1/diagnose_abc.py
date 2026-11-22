#!/usr/bin/env python3
"""Frozen Stage1/reference vs Stage2/reference vs Stage2/planner diagnosis.

Paper task semantics and the original evaluator's fixed-cohort success test are
retained. Every batch starts with the same actual state and world goal draws.
Thereafter each variant has its own native environment RNG stream and native
goal update. Policy-only RNG (notably eta=1's unused Bernoulli draw) is isolated.
This is a paired diagnostic, not a replacement for the formal pipeline gates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


def array_digest(values):
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = np.ascontiguousarray(value)
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(value.shape).encode())
        result.update(value.tobytes())
    return result.hexdigest()


def compare_initial_states(reference, other):
    """Fail on world-goal/phase/history mismatch even when qpos is identical."""
    if reference.keys() != other.keys():
        raise ValueError("Initial state fields differ")
    differences = {}
    for name in reference:
        a, b = np.asarray(reference[name]), np.asarray(other[name])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite actual initial state field: " + name)
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            differences[name] = {"shape_a": list(a.shape), "shape_b": list(b.shape)}
            if a.shape == b.shape:
                differences[name]["max_abs"] = float(np.max(np.abs(a - b)))
    if differences:
        raise ValueError("Actual initial states are not paired: " + json.dumps(differences))


def tensor_storage(value):
    """Unwrap mjlab TorchArray without silently skipping its state writes."""
    return getattr(value, "_tensor", value)


def actor_reference(runner, observations):
    """Explicit bypass: B must never call get_inference_policy/plan_contacts."""
    return runner.ac.act_inference(runner.normalizers["policy"](observations))


def local_targets_to_world(targets, valid, pelvis, yaw):
    targets = np.asarray(targets)
    c, s = np.cos(yaw), np.sin(yaw)
    result = targets.copy()
    result[..., 0] = c[..., None] * targets[..., 0] - s[..., None] * targets[..., 1]
    result[..., 1] = s[..., None] * targets[..., 0] + c[..., None] * targets[..., 1]
    result += np.asarray(pelvis)[..., None, :]
    return np.where(np.asarray(valid)[..., None], result, np.nan)


def signed_target_error(reference, ref_valid, planned, plan_valid):
    both = np.asarray(ref_valid) & np.asarray(plan_valid)
    return np.where(both[..., None], np.asarray(planned) - np.asarray(reference), np.nan), both


def first_divergence(rows, foot_indices=(3, 4)):
    """Find B/C divergence and bracket it by reference left-foot contact onsets."""
    results = []
    for clip in np.unique(rows["clip_index"]):
        for episode in np.unique(rows["episode"][rows["clip_index"] == clip]):
            ids = {}
            for variant in (0, 1, 2):
                mask = ((rows["clip_index"] == clip) & (rows["episode"] == episode) & (rows["variant"] == variant))
                ids[variant] = {int(rows["tick"][i]): i for i in np.flatnonzero(mask)}
            common = sorted(set(ids[0]) & set(ids[1]) & set(ids[2]))
            if not common:
                continue
            def earliest(test):
                return next((tick for tick in common if test(ids[1][tick], ids[2][tick])), None)
            action_tick = earliest(lambda b, c: np.max(np.abs(rows["actions"][b] - rows["actions"][c])) > 1e-5)
            xy_tick = earliest(lambda b, c: np.linalg.norm(rows["pelvis_pos_w"][b, :2] - rows["pelvis_pos_w"][c, :2]) > .01)
            contact_tick = earliest(lambda b, c: np.any(rows["measured_contact"][b, list(foot_indices)] != rows["measured_contact"][c, list(foot_indices)]))
            goal_tick = earliest(lambda b, c: not np.array_equal(rows["world_goal"][b], rows["world_goal"][c]))
            xy_over = [np.linalg.norm(rows["pelvis_pos_w"][ids[1][tick], :2] - rows["pelvis_pos_w"][ids[2][tick], :2]) > .05 for tick in common]
            clear_xy = next((common[i] for i in range(len(common) - 4) if all(xy_over[i:i + 5]) and common[i + 4] == common[i] + 4), None)
            target_error_tick = next((tick for tick in common if np.any(np.linalg.norm(rows["target_error_xyz"][ids[2][tick], list(foot_indices), :2], axis=-1) > .10)), None)
            a_ids = [ids[0][tick] for tick in common]
            contacts = rows["reference_contact_flags"][a_ids, foot_indices[0]].astype(bool)
            onsets = [common[i] for i in range(1, len(common)) if contacts[i] and not contacts[i - 1]]
            event = clear_xy if clear_xy is not None else contact_tick
            start = max([0] + [tick for tick in onsets if event is not None and tick <= event])
            end = next((tick for tick in onsets if event is not None and tick > event), common[-1])
            measured_cycles = {}
            for variant in (1, 2):
                for foot, foot_index in zip(("left", "right"), foot_indices):
                    contact = rows["measured_contact"][[ids[variant][tick] for tick in common], foot_index].astype(bool)
                    edges = [common[i] for i in range(1, len(common)) if contact[i] and not contact[i - 1]]
                    before = [tick for tick in edges if event is not None and tick <= event]
                    cycle_start = max(before) if before else common[0]
                    cycle_end = next((tick for tick in edges if tick > cycle_start), common[-1])
                    measured_cycles["ABC"[variant] + "_" + foot] = {"rising_edges": edges, "cycle_start_tick": cycle_start,
                        "cycle_end_tick": cycle_end, "complete_onset_to_onset": cycle_start in edges and cycle_end in edges}
            results.append({"clip_index": int(clip), "episode": int(episode), "first_action_difference_tick": action_tick,
                            "first_pelvis_xy_difference_gt_1cm_tick": xy_tick, "first_measured_foot_contact_difference_tick": contact_tick,
                            "first_clear_pelvis_xy_difference_gt_5cm_sustained_5ticks": clear_xy,
                            "first_same_state_C_planner_vs_reference_foot_xy_error_gt_10cm_both_valid": target_error_tick,
                            "measured_contact_cycle_at_clear_divergence": measured_cycles,
                            "first_world_goal_difference_tick": goal_tick, "reference_left_foot_onsets": onsets,
                            "first_divergent_cycle": {"start_tick": start, "end_tick": end,
                                                       "complete_onset_to_onset": start in onsets and end in onsets}})
    return results


def json_write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--heightmap-cache-dir", type=Path,
                        default=Path(os.environ.get("ACP_HEIGHTMAP_CACHE_DIR", "~/.cache/instinct_mj/supermimic_heightmaps")),
                        help="Host-local cache; ignores checkpoint-host cache paths")
    parser.add_argument("--clip-regex", default=".*")
    parser.add_argument("--trace-clips", default="php_cfg0080_01,php_cfg0098_00,php_cfg0002_00")
    parser.add_argument("--trace-all-clips", action="store_true")
    parser.add_argument("--trace-variants", default="ABC", choices=("A", "B", "C", "AB", "AC", "BC", "ABC"),
                        help="Limit saved traces, without changing the A/B/C evaluation")
    parser.add_argument("--trace-planner-inputs", action="store_true",
                        help="Save raw planner proprioception, goal and scan for same-state offline checks")
    parser.add_argument("--trace-training-contacts", action="store_true",
                        help="Record the actual force/point inputs used by force_head hindsight labeling")
    args = parser.parse_args()
    if args.episodes <= 0 or args.num_envs < args.episodes:
        parser.error("num-envs must cover at least one complete episode cohort")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["ACP_DATASET"] = str(Path(args.dataset).resolve())
    outdir = Path(args.output_dir).resolve()
    if outdir.exists() and any(outdir.iterdir()):
        parser.error("output-dir must be new or empty; existing artifacts are never overwritten")
    outdir.mkdir(parents=True, exist_ok=True)
    json_write(outdir / "status.json", {"status": "running", "pid": os.getpid(), "argv": sys.argv})
    try:
        run(args, outdir)
    except BaseException as error:
        json_write(outdir / "status.json", {"status": "needs_attention", "error": repr(error), "pid": os.getpid()})
        raise


def run(args, outdir):
    import torch
    import yaml
    import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as g1
    buffer_cfg = g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
    buffer_cfg.path = str(Path(args.dataset).resolve())
    buffer_cfg.metadata_yaml = str(Path(args.dataset).resolve() / "metadata.yaml")
    import instinct_mj.tasks  # noqa: F401
    from instinct_mj.rl import InstinctRlVecEnvWrapper
    from instinct_mj.rl.acp.runner import GOAL_SLICE, TARGETS_SLICE, PLANNER_PROPRIO, SCAN_DIM, clip_family
    from instinct_mj.rl.acp.sim import ACPSimReader
    from instinct_mj.rl.acp.planner import AttentionContactPlanner, CONTACT_BODIES, CONTACT_LINKS
    from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls

    torch.set_grad_enabled(False)
    device = "cuda:0"
    N, E = args.num_envs, args.episodes
    metadata = Path(args.dataset).resolve() / "metadata.yaml"
    metadata_hash = digest_file(metadata)
    executed_tool_sha256 = digest_file(__file__)

    def capture_rng():
        return {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state().clone()}

    def restore_rng(state):
        torch.set_rng_state(state["cpu"])
        torch.cuda.set_rng_state(state["cuda"])

    @contextmanager
    def rng_context(bundle):
        outer = capture_rng()
        restore_rng(bundle["rng"])
        try:
            yield
        finally:
            bundle["rng"] = capture_rng()
            restore_rng(outer)

    def normalizer_hash(runner):
        return array_digest({name: value.detach().cpu().numpy() for name, value in runner.normalizers["policy"].state_dict().items()})

    def disabled_termination(env, **kwargs):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    bundles = []
    for label, checkpoint, task in [("A", args.stage1_checkpoint, "Instinct-ACP-Track-G1-v1"),
                                    ("B", args.stage2_checkpoint, "Instinct-ACP-Stage2-G1-v1"),
                                    ("C", args.stage2_checkpoint, "Instinct-ACP-Stage2-G1-v1")]:
        torch.manual_seed(args.seed)
        cfg = load_env_cfg(task)
        cfg.seed, cfg.scene.num_envs, cfg.episode_length_s = args.seed, N, 120.0
        cfg.disable_randomization()
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            actuator.delay_max_lag = 0
        cfg.events["reset_robot"].params["randomize_velocity_range"] = {}
        for term in ("base_xy_too_far", "foot_pos_too_far", "contact_violation"):
            cfg.terminations[term].func = disabled_termination
        env = load_env_cls(task)(cfg=cfg, device=device)
        vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
        agent = load_instinct_rl_cfg(task).to_dict()
        saved_path = Path(checkpoint).resolve().parent / "params/agent.yaml"
        if not saved_path.is_file():
            raise ValueError("Checkpoint saved agent config missing: " + str(saved_path))
        saved = yaml.safe_load(saved_path.read_text())
        agent.update({key: value for key, value in saved.items() if key in agent})
        agent["heightmap_cache_dir"] = str(args.heightmap_cache_dir.expanduser().resolve())
        agent.update(compile=False, compile_policy=False, buffer_steps=64, warmup_steps=32)
        runner = load_runner_cls(task)(vec, agent, None, device)
        runner.load(checkpoint)
        runner.motion, runner.kin, runner.scanner, runner.mdp = runner.build_analytic_mdp()
        runner.reader = ACPSimReader(env)
        if args.trace_training_contacts:
            if runner.cfg["contact_detect"] != "force_head":
                raise ValueError("Training-contact tracing currently requires contact_detect=force_head")
            runner.sdf = None
            sensor = env.scene["acp_contact_points"]
            runner.contact_point_ids = [sensor.primary_names.index(name) for name in CONTACT_LINKS]
            runner.last_contact_pos = torch.zeros(N, 9, 3, device=device)
        mb = next(iter(env.scene["motion_reference"].motion_buffers.values()))
        runner.motion_names = [Path(path).stem for path in mb._all_motion_files]
        env.curriculum_manager.get_term_cfg("episode_start").func.mode = "start"
        env.cfg.auto_reset = False
        runner.eval_mode()
        policy = runner.get_inference_policy(device=device) if label == "C" else None
        if label == "A" and runner.planner is not None:
            raise ValueError("A is not a Stage1 reference-only runner")
        if label in ("B", "C") and runner.planner is None:
            raise ValueError("Stage2 checkpoint/config did not instantiate a planner")
        bundle = {"label": label, "checkpoint": str(Path(checkpoint).resolve()), "task": task, "env": env, "vec": vec,
                  "runner": runner, "mb": mb, "policy": policy, "rng": capture_rng(), "normalizer_sha256": normalizer_hash(runner)}
        bundles.append(bundle)
        print("Loaded", label, checkpoint, "normalizer", bundle["normalizer_sha256"], flush=True)

    names = bundles[0]["runner"].motion_names
    if any(bundle["runner"].motion_names != names for bundle in bundles):
        raise ValueError("Variant motion order differs")
    K = len(names)
    selected = [k for k, name in enumerate(names) if re.search(args.clip_regex, name)]
    if not selected:
        raise ValueError("No selected clips")
    trace_names = set(names) if args.trace_all_clips else set(args.trace_clips.split(","))
    if trace_names - set(names):
        raise ValueError("Trace clip missing: " + repr(trace_names - set(names)))
    families = [clip_family(item.get("group", "")) for item in bundles[0]["mb"].yaml_data["motion_files"]]
    if any(families[k] == "get_up" for k in selected):
        raise ValueError("This fixed flat locomotion diagnosis does not implement get_up success semantics")

    def cpu(value):
        return value.detach().cpu().numpy()

    def tensor_view(value):
        # mjlab WarpBridge exposes TorchArray proxies, not Tensor subclasses.
        value = tensor_storage(value)
        return value if isinstance(value, torch.Tensor) else None

    physics_hashes = {}
    for bundle in bundles:
        model = bundle["env"].sim.mj_model
        values = {name: np.asarray(getattr(model, name)) for name in
            ("body_mass", "body_ipos", "body_inertia", "dof_armature", "dof_damping", "actuator_gainprm", "actuator_biasprm", "geom_friction", "jnt_range")}
        for name in values.copy():
            value = tensor_view(getattr(bundle["env"].sim.model, name, None))
            if isinstance(value, torch.Tensor):
                values["batched_" + name] = cpu(value)
        physics_hashes[bundle["label"]] = array_digest(values)
    if len(set(physics_hashes.values())) != 1:
        raise ValueError("Paper physics parameters differ across variants: " + repr(physics_hashes))

    def state_snapshot(bundle, ids):
        env, runner = bundle["env"], bundle["runner"]
        f, phase, origin = runner.motion.current_frame()
        obs, extras = bundle["vec"].get_observations()
        goal = runner.reader.goal_term
        result = {"qpos": cpu(env.sim.data.qpos[ids]), "qvel": cpu(env.sim.data.qvel[ids]),
                "policy_observation": cpu(obs[ids]), "planner_observation": cpu(extras["observations"]["planner"][ids]),
                "motion_frame": cpu(f[ids]), "motion_phase": cpu(phase[ids]), "motion_origin": cpu(origin[ids]),
                "world_goal": cpu(torch.cat((goal.target_pos_w, goal.target_yaw[:, None], goal.target_time[:, None]), -1)[ids]),
                "last_action": cpu(env.action_manager.action[ids]), "control_dt": np.full((len(ids), 1), env.step_dt)}
        for name in ("qacc_warmstart", "act", "ctrl", "qfrc_applied", "xfrc_applied", "time"):
            value = tensor_view(getattr(env.sim.data, name, None))
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == N:
                result["mj_" + name] = cpu(value[ids])
        for name in ("_prev_action", "_prev_prev_action"):
            result[name] = cpu(getattr(env.action_manager, name)[ids])
        for name in ("_raw_actions", "_processed_actions"):
            value = getattr(env.action_manager.get_term("joint_pos"), name)
            result[name] = cpu(value[ids])
        return result

    def canonicalize_initial_state(source, destination, ids):
        """Control actual initial physics; retain A's native state without alteration.

        Independent motion loading can differ by a few float32 ulps. Do not hide
        that with a relaxed pairing predicate: copy A's actual state, controls and
        warmstart, then rebuild B/C's reset observations without advancing time.
        """
        before = state_snapshot(destination, ids)
        expected = state_snapshot(source, ids)
        for name in ("qpos", "motion_frame", "motion_phase", "motion_origin", "world_goal", "planner_observation"):
            compare_initial_states({name: expected[name]}, {name: before[name]})
        qvel_delta = float(np.max(np.abs(before["qvel"] - expected["qvel"])))
        if qvel_delta > 1e-4:
            raise ValueError("Unexpected native initial qvel discrepancy: " + str(qvel_delta))
        a, b = source["env"], destination["env"]
        copied = []
        for name in ("_action", "_prev_action", "_prev_prev_action"):
            getattr(b.action_manager, name).copy_(getattr(a.action_manager, name))
            copied.append("action_manager." + name)
        for name in ("_raw_actions", "_processed_actions"):
            getattr(b.action_manager.get_term("joint_pos"), name).copy_(getattr(a.action_manager.get_term("joint_pos"), name))
            copied.append("joint_pos_action." + name)
        for name in ("joint_pos_target", "joint_vel_target", "joint_effort_target"):
            getattr(b.scene["robot"].data, name).copy_(getattr(a.scene["robot"].data, name))
            copied.append("robot.data." + name)
        for name in ("qpos", "qvel", "qacc_warmstart", "act", "ctrl", "qfrc_applied", "xfrc_applied", "time"):
            av, bv = tensor_view(getattr(a.sim.data, name, None)), tensor_view(getattr(b.sim.data, name, None))
            if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor) and av.shape == bv.shape:
                bv.copy_(av)
                copied.append("mj_data." + name)
        b.sim.forward()
        b.command_manager.compute(dt=0.0, env_ids=ids)
        b.sim.sense()
        # Match the reset contact-history sample, not a new physical time step.
        a_contact = a.scene["contact_forces"].data
        b_contact = b.scene["contact_forces"].data
        for name in set(getattr(a_contact, "__dict__", {})) | {"force_history", "force"}:
            av = getattr(a_contact, name, None)
            bv = getattr(b_contact, name, None)
            if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor) and av.shape == bv.shape:
                bv.copy_(av)
                copied.append("contact_sensor." + name)
        # Forward may refresh a solver warmstart; it is an input to the NEXT step.
        for name in ("qacc_warmstart", "act", "ctrl"):
            av, bv = tensor_view(getattr(a.sim.data, name, None)), tensor_view(getattr(b.sim.data, name, None))
            if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
                bv.copy_(av)
        b.observation_manager.reset(env_ids=ids)
        b.obs_buf = b.observation_manager.compute(update_history=True, env_ids=ids)
        destination["obs"] = destination["vec"].get_observations()[0]
        after = state_snapshot(destination, ids)
        compare_initial_states(expected, after)
        for name in ("motion_frame", "motion_phase", "motion_origin", "world_goal"):
            compare_initial_states({name: before[name]}, {name: after[name]})
        return {"variant": destination["label"], "native_before_sha256": array_digest(before), "canonical_after_sha256": array_digest(after),
                "native_qpos_sha256": array_digest({"qpos": before["qpos"]}), "native_qvel_sha256": array_digest({"qvel": before["qvel"]}),
                "canonical_qpos_sha256": array_digest({"qpos": after["qpos"]}), "canonical_qvel_sha256": array_digest({"qvel": after["qvel"]}),
                "source_A_qpos_sha256": array_digest({"qpos": expected["qpos"]}), "source_A_qvel_sha256": array_digest({"qvel": expected["qvel"]}),
                "native_qvel_max_abs_difference": qvel_delta, "copied_state_fields": copied, "reference_and_goal_not_advanced": True}

    def reset_cohort(bundle, ids, clip):
        bundle["mb"]._motion_weights[:] = 0
        bundle["mb"]._motion_weights[clip] = 1
        bundle["env"].reset(env_ids=ids)

    for bundle in bundles:
        term_names = [name for name in bundle["env"].termination_manager.active_terms if name not in ("time_out", "dataset_exhausted")]
        bundle.update(survive=torch.zeros(K, device=device), success=torch.zeros(K, device=device), strict=torch.zeros(K, device=device),
                      agree_n=torch.zeros(K, device=device), agree_d=torch.zeros(K, device=device), count=torch.zeros(K, device=device), unfinished=torch.zeros(K, device=device),
                      term_names=term_names, causes=torch.zeros(len(term_names) + 1, K, device=device), final_xy=[[] for _ in names], max_xy=[[] for _ in names], onsets=[[] for _ in names])

    trace = {}
    initial_arrays = {}
    pairing = []
    goal_divergences = []
    first_goal_seen = set()
    policy_match_max = {"A": 0.0, "B": 0.0, "C": 0.0}
    started = time.monotonic()

    def append_arrays(store, values):
        for key, value in values.items():
            store.setdefault(key, []).append(np.asarray(value))

    def trace_frame(bundle, clip_ids, active, tick, actions):
        if bundle["label"] not in args.trace_variants:
            return None
        runner, env = bundle["runner"], bundle["env"]
        trace_ids = torch.tensor([i for i in range(N) if bool(active[i]) and names[int(clip_ids[i])] in trace_names], device=device, dtype=torch.long)
        if not len(trace_ids):
            return None
        obs, extras = bundle["vec"].get_observations()
        points = extras["observations"]["planner"].view(-1, SCAN_DIM, 3)
        with torch.random.fork_rng(devices=[0]):
            output = bundles[2]["runner"].planner(obs[:, :PLANNER_PROPRIO], obs[:, GOAL_SLICE], points)
            planned, planned_valid = AttentionContactPlanner.decode(output, points)
            replaced = obs.clone()
            replaced[:, TARGETS_SLICE] = planned.flatten(1)
            counter_actions = torch.clamp(torch.stack((actor_reference(bundles[0]["runner"], obs), actor_reference(bundles[1]["runner"], obs), actor_reference(bundles[2]["runner"], replaced)), 1), -20, 20)
        index = ord(bundle["label"]) - ord("A")
        error = float((counter_actions[trace_ids, index] - actions[trace_ids]).abs().max())
        policy_match_max[bundle["label"]] = max(policy_match_max[bundle["label"]], error)
        if error > 3e-5:
            raise ValueError("Trace counterfactual does not match actual policy " + bundle["label"] + ": " + str(error))
        reference, ref_valid = runner.reader.contacts.targets(estimated=True)
        S = runner.reader.state()
        f, phase, origin = runner.motion.current_frame()
        reference_S = runner.mdp.reference_state(f, origin)
        yaw = torch.atan2(S[1][:, 1, 0], S[1][:, 0, 0])
        ref_yaw = torch.atan2(reference_S[1][:, 1, 0], reference_S[1][:, 0, 0])
        goal = runner.reader.goal_term
        ref, rv, plan, pv, pos, yaw_np = (cpu(value[trace_ids]) for value in (reference, ref_valid, planned, planned_valid, S[0], yaw))
        errors, both = signed_target_error(ref, rv, plan, pv)
        count = len(trace_ids)
        append_arrays(trace, {"variant": np.full(count, index, np.int8), "clip_index": cpu(clip_ids[trace_ids]),
                             "episode": cpu(trace_ids % E), "tick": np.full(count, tick, np.int32), "time_s": np.full(count, tick * env.step_dt),
                             "pelvis_pos_w": pos, "pelvis_yaw": yaw_np, "ref_pelvis_pos_w": cpu(reference_S[0][trace_ids]), "ref_pelvis_yaw": cpu(ref_yaw[trace_ids]),
                             "motion_frame": cpu(f[trace_ids]), "motion_phase": cpu(phase[trace_ids]), "motion_origin": cpu(origin[trace_ids]),
                             "world_goal": cpu(torch.cat((goal.target_pos_w, goal.target_yaw[:, None], goal.target_time[:, None]), -1)[trace_ids]), "local_goal": cpu(obs[trace_ids, GOAL_SLICE]),
                             "ref_targets_local": ref, "ref_targets_world": local_targets_to_world(ref, rv, pos, yaw_np), "ref_valid": rv,
                             "planner_targets_local": plan, "planner_targets_world": local_targets_to_world(plan, pv, pos, yaw_np), "planner_valid": pv,
                             "planner_occupancy": cpu(torch.sigmoid(output["occ_logit"])[trace_ids]), "target_error_xyz": errors,
                             "target_error_both_valid": both, "foot_target_error_xyz": errors[:, [3, 4]],
                             "actions": cpu(actions[trace_ids]), "same_state_counterfactual_actions": cpu(counter_actions[trace_ids]),
                             "measured_contact": cpu(runner.reader.contact()[trace_ids]), "reference_contact_flags": cpu(runner.motion.flag(f, phase)[trace_ids]),
                             "active": np.ones(count, bool)})
        if args.trace_planner_inputs:
            append_arrays(trace, {"planner_proprio": cpu(obs[trace_ids, :PLANNER_PROPRIO]),
                                  "planner_points": cpu(points[trace_ids])})
        if args.trace_training_contacts:
            link_pos, link_rot = runner.kin.forward(*S)[:2]
            force_z, contact_pos = runner.contact_measurement(link_pos[:, :9], env.episode_length_buf == 0, link_rot[:, :9])
            scanner = env.scene["height_scanner"]
            quat = scanner.estimated_quat_w
            est_yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                                  1 - 2 * (quat[:, 2].square() + quat[:, 3].square()))
            append_arrays(trace, {"training_contact_force_z": cpu(force_z[trace_ids]),
                                  "training_contact_pos_w": cpu(contact_pos[trace_ids]),
                                  "training_contact_link_pos_w": cpu(link_pos[trace_ids, :9]),
                                  "training_contact_link_rot_w": cpu(link_rot[trace_ids, :9]),
                                  "reference_contact_link_pos_w": cpu((runner.motion.link_p[f] + origin[:, None])[trace_ids, :9]),
                                  "reference_contact_link_rot_w": cpu(runner.motion.link_R[f][trace_ids, :9]),
                                  "training_estimated_pelvis_pos_w": cpu(scanner.estimated_pos_w[trace_ids]),
                                  "training_estimated_pelvis_yaw": cpu(est_yaw[trace_ids])})
        return trace_ids

    per_batch = N // E
    for first in range(0, len(selected), per_batch):
        batch = selected[first:first + per_batch]
        valid_count = len(batch) * E
        ids_all = torch.arange(valid_count, device=device)
        shared_reset_rng = copy.deepcopy(bundles[0]["rng"])
        snapshots = []
        canonical_receipts = []
        for bundle in bundles:
            bundle["rng"] = copy.deepcopy(shared_reset_rng)
            with rng_context(bundle):
                env_clip = torch.full((N,), batch[0], device=device, dtype=torch.long)
                active = torch.zeros(N, device=device, dtype=torch.bool)
                for i, clip in enumerate(batch):
                    ids = torch.arange(i * E, (i + 1) * E, device=device)
                    env_clip[ids], active[ids] = clip, True
                    reset_cohort(bundle, ids, clip)
                if valid_count < N:
                    reset_cohort(bundle, torch.arange(valid_count, N, device=device), batch[0])
                bundle.update(env_clip=env_clip, active=active, obs=bundle["vec"].get_observations()[0], maxdev=torch.zeros(N, device=device),
                              maxxy=torch.zeros(N, device=device), prev_contact=torch.zeros(N, 9, device=device))
                if bundle["label"] != "A":
                    # Even noise=0 observation terms draw rand/randn. Rebuilding
                    # a canonical initial history must not consume env RNG.
                    with torch.random.fork_rng(devices=[0]):
                        canonical_receipts.append(canonicalize_initial_state(bundles[0], bundle, ids_all))
                snapshot = state_snapshot(bundle, ids_all)
                snapshots.append(snapshot)
                row_count = valid_count
                append_arrays(initial_arrays, snapshot | {"variant": np.full(row_count, ord(bundle["label"]) - 65, np.int8),
                                                         "clip_index": cpu(env_clip[ids_all]), "episode": cpu(ids_all % E)})
        compare_initial_states(snapshots[0], snapshots[1])
        compare_initial_states(snapshots[0], snapshots[2])
        if any(not torch.equal(bundle["rng"][key], bundles[0]["rng"][key]) for bundle in bundles[1:] for key in ("cpu", "cuda")):
            raise ValueError("Post-reset environment RNG states differ across variants")
        pairing.append({"batch": first // per_batch, "clips": [names[k] for k in batch], "episodes": valid_count,
                        "actual_initial_state_equal": True, "state_sha256": array_digest(snapshots[0]),
                        "post_reset_rng_equal": True,
                        "post_reset_rng_sha256": {key: hashlib.sha256(bundles[0]["rng"][key].cpu().numpy().tobytes()).hexdigest() for key in ("cpu", "cuda")},
                        "canonicalization": canonical_receipts,
                        "rng_reset_sha256": hashlib.sha256(shared_reset_rng["cuda"].cpu().numpy().tobytes()).hexdigest()})
        for tick in range(args.max_steps):
            if not any(bool(bundle["active"].any()) for bundle in bundles):
                break
            pre_goals = []
            pre_active = []
            for bundle in bundles:
                goal = bundle["runner"].reader.goal_term
                pre_goals.append(cpu(torch.cat((goal.target_pos_w, goal.target_yaw[:, None], goal.target_time[:, None]), -1)))
                pre_active.append(cpu(bundle["active"]))
            for variant in (1, 2):
                same = np.all(pre_goals[0] == pre_goals[variant], axis=-1)
                for row in np.flatnonzero(~same & pre_active[0] & pre_active[variant]):
                    key = (int(bundles[0]["env_clip"][row]), int(row % E), variant)
                    if key not in first_goal_seen:
                        first_goal_seen.add(key)
                        goal_divergences.append({"clip": names[key[0]], "episode": key[1], "variant": "ABC"[variant], "tick": tick,
                                                 "world_goal_A": pre_goals[0][row].tolist(), "world_goal_variant": pre_goals[variant][row].tolist()})
            for bundle in bundles:
                if not bool(bundle["active"].any()):
                    continue
                with rng_context(bundle):
                    runner, env, active, env_clip = bundle["runner"], bundle["env"], bundle["active"], bundle["env_clip"]
                    S0 = runner.reader.state()
                    with torch.random.fork_rng(devices=[0]):
                        actions = bundle["policy"](bundle["obs"]) if bundle["label"] == "C" else actor_reference(runner, bundle["obs"])
                    # Original paper evaluator consumes this draw even though its noise multiplier is zero.
                    actions = torch.clamp(actions + 0.0 * torch.randn_like(actions), -20, 20)
                    traced_ids = trace_frame(bundle, env_clip, active, tick, actions)
                    bundle["obs"], _, _, _ = bundle["vec"].step(actions)
                    S1 = runner.reader.state()
                    bad = ~torch.stack([torch.isfinite(value).flatten(1).all(-1) for value in S1]).all(0)
                    S1 = tuple(torch.where(bad.view(-1, *[1] * (s0.dim() - 1)), s0, s1) for s0, s1 in zip(S0, S1))
                    f, phase, origin = runner.motion.current_frame()
                    c = runner.reader.contact()
                    bundle["maxdev"] = torch.maximum(bundle["maxdev"], runner.mdp.link_deviation(S1, f, origin))
                    xy = (S1[0] - runner.mdp.reference_state(f, origin)[0])[:, :2].norm(dim=-1)
                    bundle["maxxy"] = torch.maximum(bundle["maxxy"], xy)
                    link_error = (runner.mdp.kin.forward(*S1)[0] - (runner.motion.link_p[f] + origin[:, None])).norm(dim=-1)
                    new = (c > .5) & (bundle["prev_contact"] < .5) & active[:, None]
                    bundle["agree_n"].index_add_(0, env_clip, (c == runner.motion.flag(f, phase).float()).float().sum(-1) * active.float())
                    bundle["agree_d"].index_add_(0, env_clip, 9.0 * active.float())
                    for clip in env_clip[new.any(-1)].unique().tolist():
                        rows = (env_clip == clip) & new.any(-1)
                        bundle["onsets"][clip].append(link_error[rows][new[rows]].cpu())
                    bundle["prev_contact"] = c.clone()
                    tm = env.termination_manager
                    done = tm.dones | bad
                    if traced_ids is not None:
                        failure = env.reset_terminated | bad
                        reached_trace = (tm.get_term("dataset_exhausted") | tm.get_term("time_out")) & ~failure
                        append_arrays(trace, {"done_after_step": cpu(done[traced_ids]), "failed_after_step": cpu(failure[traced_ids]),
                                              "success_after_step": cpu((done & reached_trace & (xy < .5))[traced_ids]),
                                              "next_pelvis_pos_w": cpu(S1[0][traced_ids]),
                                              "next_pelvis_yaw": cpu(torch.atan2(S1[1][:, 1, 0], S1[1][:, 0, 0])[traced_ids])})
                    if done.any():
                        ids = done.nonzero().squeeze(-1)
                        counted = ids[active[ids]]
                        if len(counted):
                            failed = env.reset_terminated[counted] | bad[counted]
                            reached = (tm.get_term("dataset_exhausted")[counted] | tm.get_term("time_out")[counted]) & ~failed
                            clip = env_clip[counted]
                            success = reached & (xy[counted] < .5)
                            bundle["count"].index_add_(0, clip, torch.ones(len(counted), device=device))
                            bundle["survive"].index_add_(0, clip, reached.float())
                            bundle["success"].index_add_(0, clip, success.float())
                            bundle["strict"].index_add_(0, clip, (success & (bundle["maxdev"][counted] < .35)).float())
                            fired = torch.stack([tm.get_term(name)[counted] for name in bundle["term_names"]] + [bad[counted]]).float()
                            bundle["causes"].index_add_(1, clip, fired)
                            for clip_id, maximum, final in zip(clip.tolist(), bundle["maxxy"][counted].tolist(), xy[counted].tolist()):
                                bundle["max_xy"][clip_id].append(maximum)
                                bundle["final_xy"][clip_id].append(final)
                            active[counted] = False
                        reset_cohort(bundle, ids, batch[0])
                        bundle["obs"] = bundle["vec"].get_observations()[0]
                        bundle["maxdev"][ids], bundle["maxxy"][ids], bundle["prev_contact"][ids] = 0, 0, 0
        for bundle in bundles:
            active = bundle["active"]
            if bool(active.any()):
                bundle["unfinished"].index_add_(0, bundle["env_clip"][active], torch.ones(int(active.sum()), device=device))
                bundle["count"].index_add_(0, bundle["env_clip"][active], torch.ones(int(active.sum()), device=device))
        print("Batch", first // per_batch + 1, "of", (len(selected) + per_batch - 1) // per_batch, "ticks", tick + 1,
              "counts", {b["label"]: int(b["count"].sum()) for b in bundles}, flush=True)

    reports = {}
    for bundle in bundles:
        rows = []
        for k in selected:
            count = int(bundle["count"][k])
            onset = torch.cat(bundle["onsets"][k]) if bundle["onsets"][k] else torch.zeros(0)
            safe = lambda value: float(value) if len(onset) else None
            rows.append({"clip": names[k], "family": families[k], "episodes": count, "unfinished": int(bundle["unfinished"][k]),
                         "survive": float(bundle["survive"][k] / max(1, count)), "success": float(bundle["success"][k] / max(1, count)),
                         "strict_success": float(bundle["strict"][k] / max(1, count)), "label_agreement": float(bundle["agree_n"][k] / bundle["agree_d"][k].clamp(min=1)),
                         "max_xy_median": float(np.median(bundle["max_xy"][k])) if bundle["max_xy"][k] else None,
                         "onset_err_median": safe(onset.median()) if len(onset) else None,
                         "onset_err_p90": safe(torch.quantile(onset, .9)) if len(onset) else None, "onsets": len(onset)})
        overall = {key: sum(row[key] for row in rows) / len(rows) for key in ("survive", "success", "strict_success", "label_agreement")}
        episode_count = sum(row["episodes"] for row in rows)
        term_counts = {name: int(bundle["causes"][i].sum()) for i, name in enumerate(bundle["term_names"] + ["nonfinite"])}
        if normalizer_hash(bundle["runner"]) != bundle["normalizer_sha256"]:
            raise ValueError("Normalizer changed during frozen diagnosis")
        report = {"checkpoint": bundle["checkpoint"], "task": bundle["task"], "mode": "paper", "seed": args.seed,
                  "metadata_path": str(metadata), "metadata_sha256": metadata_hash, "overall": overall, "clips": rows,
                  "episode_count": episode_count, "termination_counts_may_overlap": term_counts,
                  "max_xy_episodes": {names[k]: bundle["max_xy"][k] for k in selected},
                  "diagnostic_only": True, "contact_source": "planner" if bundle["label"] == "C" else "reference_explicit_actor_bypass",
                  "normalizer_sha256": bundle["normalizer_sha256"], "checkpoint_sha256": digest_file(bundle["checkpoint"])}
        json_write(outdir / (bundle["label"] + ".paper.json"), report)
        reports[bundle["label"]] = report
    schema = {"variant_names": ["A_stage1_reference", "B_stage2_reference_bypass", "C_stage2_native_planner"], "clip_names": names,
              "contact_body_names": list(CONTACT_BODIES), "foot_indices_left_right": [3, 4],
              "sample_time": "Pre-action state at tick; actions are applied over [tick,tick+1]. Rows only exist while that episode is active.",
              "target_error": "planner minus reference in the same current pelvis yaw frame; XYZ=forward,left,up; NaN unless both valid.",
              "world_goal": ["x", "y", "z", "yaw", "clip_local_target_time_seconds"],
              "same_state_counterfactual_actions": "[A actor with A normalizer/reference targets, B actor with B normalizer/reference targets, C actor with C normalizer/planner targets], all evaluated on this row's SAME raw state.",
              "diagnostic_divergence_thresholds_not_gates": {"numerical_action_max_abs": 1e-5, "small_pelvis_xy_m": .01,
                  "clear_pelvis_xy_m": .05, "clear_pelvis_xy_consecutive_ticks": 5, "same_state_foot_target_xy_m_both_valid": .10,
                  "gait_cycle": "Measured left/right foot rising edges, separately for B and C; complete flag false when sequence boundaries censor an edge."},
              "goal_semantics": "Original native goal updater and U(.5,1.5) sampling retained. Only batch initial RNG/actual states are paired; later world goals may diverge through native reset/RNG history. Local goals always use each variant's own pelvis pose.",
              "scope": "Training metadata only; holdout is never loaded; no learning or sidecar mutations."}
    schema["traced_variants"] = list(args.trace_variants)
    schema["planner_inputs_recorded"] = args.trace_planner_inputs
    if args.trace_training_contacts:
        schema["training_contact_trace"] = "Exact force_head contact_measurement() inputs, strongest solver contact point with last-point fallback, pre-action; use label_episode() defaults for hindsight labels. Estimated pelvis pose is the scanner pose used by training."
    initial_values = {key: np.concatenate(value, axis=0) for key, value in initial_arrays.items()}
    np.savez_compressed(outdir / "initial_states.npz", **initial_values)
    if trace:
        trace_values = {key: np.concatenate(value, axis=0) for key, value in trace.items()}
        np.savez_compressed(outdir / "trace_selected.npz", **trace_values)
        json_write(outdir / "first_divergence.json", first_divergence(trace_values))
    json_write(outdir / "trace_selected.schema.json", schema)
    json_write(outdir / "pairing.json", {"all_initial_states_equal": True, "metadata_sha256": metadata_hash, "batches": pairing,
                                         "protocol_version": 2, "canonicalization_rng_isolated": True, "all_post_reset_rng_states_equal": True,
                                         "paper_physics_parameter_sha256": physics_hashes,
                                         "first_native_world_goal_divergences": goal_divergences, "policy_rng_isolated": True,
                                         "actual_vs_counterfactual_action_max_abs": policy_match_max,
                                         "notes": ["B explicitly calls actor(normalizer(raw_reference_obs)); no planner replacement.",
                                                   "A environment RNG continues exactly within its own evaluator loop; each B/C batch starts at A's reset RNG state.",
                                                   "Environment RNG thereafter advances independently using native goal updates and partial resets; no world/local target schedule is forced.",
                                                   "C eta=1 Bernoulli policy RNG draws are forked and discarded, preventing irrelevant policy RNG from perturbing environment goal sampling."]})
    provenance = {"argv": sys.argv, "source_tool_sha256": executed_tool_sha256, "metadata_sha256": metadata_hash,
                  "protocol_version": 2, "canonicalization_rng_isolated": True,
                  "num_envs_per_variant": N, "episodes_per_window": E, "seed": args.seed, "max_steps": args.max_steps,
                  "scope": [names[k] for k in selected], "control_dt": bundles[0]["env"].step_dt,
                  "checkpoint_sha256": {b["label"]: digest_file(b["checkpoint"]) for b in bundles},
                  "normalizer_sha256": {b["label"]: b["normalizer_sha256"] for b in bundles}, "elapsed_seconds": time.monotonic() - started,
                  "thresholds": {"xy_end": .5, "strict_eps": .35, "paper_terminations": "height/tilt only; native illegal-reset timeout retained"}}
    try:
        provenance["source_git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
        provenance["source_git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=Path(__file__).resolve().parents[2], text=True)
    except subprocess.CalledProcessError:
        pass
    json_write(outdir / "provenance.json", provenance)
    dataset_provenance = json.loads((Path(args.dataset) / "provenance.json").read_text())
    source_by_name = {Path(item["motion_file"]).stem: item["source_motion_sha256"] for item in dataset_provenance["selected_windows"]}
    macros = {}
    for label, report in reports.items():
        grouped = {}
        for row in report["clips"]:
            grouped.setdefault(source_by_name[row["clip"]], []).append(row["success"])
        per_source = {key: sum(values) / len(values) for key, values in grouped.items()}
        macros[label] = {"source_macro_success": sum(per_source.values()) / len(per_source), "source_count": len(per_source),
                         "window_count": len(report["clips"]), "clip_mean_success": report["overall"]["success"],
                         "full_training_cohort": len(selected) == K, "diagnostic_only": True}
        if len(selected) == K:
            import pipeline
            gate = pipeline.tracker_gate(outdir / (label + ".paper.json"), args.dataset, "train", report["checkpoint"], E, "paper")
            macros[label].update(gate)
            macros[label].update(threshold=.9, passed=gate["source_macro_success"] >= .9)
    json_write(outdir / "source_macros.json", macros)
    json_write(outdir / "status.json", {"status": "complete", "episode_counts": {k: v["episode_count"] for k, v in reports.items()},
                                       "clip_mean_success": {k: v["overall"]["success"] for k, v in reports.items()}, "elapsed_seconds": time.monotonic() - started,
                                       "initial_pairing_passed": True, "normalizers_unchanged": True})
    for label, report in reports.items():
        print(label, json.dumps(report["overall"]), "episodes", report["episode_count"], flush=True)


if __name__ == "__main__":
    main()
