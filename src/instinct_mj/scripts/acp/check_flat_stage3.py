"""Validate the optional flat Stage 3 task without training or robot I/O.

CPU checks cover config isolation, seeded reset sampling, distance/heading
bounds, actual MuJoCo rays, and reward parameters. ``--live`` adds four GPU
envs and three zero-action steps to check the native 939-input interface.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

import instinct_mj.tasks  # noqa: F401
import instinct_mj.tasks.acp.config.g1.acp_tracking_cfg as g1
from instinct_mj.tasks.acp.goal_terrains import NUM_LEVELS, TERRAIN_TYPES
from instinct_mj.tasks.acp.mdp.goal import AcpStage3GoalCommand
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls

TASK = "Instinct-ACP-Stage3-Flat-G1-v1"
PLAY = "Instinct-ACP-Stage3-Flat-G1-Play-v1"
ORIGINAL = "Instinct-ACP-Stage3-G1-v1"


def check_config():
    original, cfg, play = load_env_cfg(ORIGINAL), load_env_cfg(TASK), load_env_cfg(PLAY)
    assert len(TERRAIN_TYPES) == 17 and "flat" not in TERRAIN_TYPES
    assert original.scene.arena.types == TERRAIN_TYPES
    for selected in (cfg, play):
        assert selected.scene.arena_cfg.types == selected.scene.arena.types == ("flat",)
        assert selected.scene.arena.num_types == 1
        goal = selected.commands["acp_goal"]
        assert goal.reset_yaw_range == math.pi and goal.goal_min_distance == .5
        rewards = selected.rewards["rewards"]
        assert rewards["move2goal"].params["radius"] == .2
        assert rewards["standatgoal"].params["radius"] == .2
        assert rewards["standatgoal"].params["heading_tolerance"] == .25
        for name, term in original.rewards["rewards"].items():
            assert rewards[name].func is term.func and rewards[name].weight == term.weight
    assert original.commands["acp_goal"].reset_yaw_range == .4
    assert original.rewards["rewards"]["move2goal"].params["radius"] == .5
    assert original.rewards["rewards"]["standatgoal"].params["heading_tolerance"] == .5
    assert cfg.commands["acp_goal"].goal_env_fraction == .5
    assert play.commands["acp_goal"].goal_env_fraction == 1.
    assert cfg.scene.num_envs == 2048 and play.scene.num_envs == 1
    assert load_runner_cls(TASK) is load_runner_cls(ORIGINAL)
    assert load_instinct_rl_cfg(TASK).experiment_name == "g1AcpStage3FlatV1"
    return cfg


def check_sampling_and_rays(cfg):
    arena, count = cfg.scene.arena, 1024
    tables = arena.tables("cpu")
    assert tables.num_tiles == NUM_LEVELS
    for flag in ("foot", "stumble", "slip", "heading_any"):
        assert getattr(tables, flag).all()
    assert not tables.getup.any() and not tables.lying.any() and torch.all(tables.z_low == 0)
    tiles = torch.arange(NUM_LEVELS).repeat_interleave(count)
    n = len(tiles)
    fake = SimpleNamespace(tile_id=tiles, tables=tables, device="cpu", cfg=cfg.commands["acp_goal"],
        target_pos_w=torch.zeros(n, 3), target_yaw=torch.zeros(n), target_time=torch.zeros(n))
    torch.manual_seed(2026)
    spawn, face, lying, _ = AcpStage3GoalCommand.sample_reset(fake, torch.arange(n))
    goals, headings = fake.target_pos_w.clone(), fake.target_yaw.clone()
    distance = torch.linalg.vector_norm(goals[:, :2]-spawn[:, :2], dim=-1)
    assert distance.min() > 1. and distance.max() < 4.
    assert torch.all(spawn[:, 2] == 0) and torch.allclose(goals[:, 2], torch.full((n,), .78))
    assert face.min() < -3. and face.max() > 3. and headings.min() < -3. and headings.max() > 3.
    assert not lying.any()
    torch.manual_seed(2026)
    repeat, repeat_face, _, _ = AcpStage3GoalCommand.sample_reset(fake, torch.arange(n))
    assert torch.equal(repeat, spawn) and torch.equal(repeat_face, face) and torch.equal(fake.target_pos_w, goals)
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody><body name="terrain"/></worldbody></mujoco>')
    arena.add_to_spec(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    worst = 0.
    ray_count = 0
    hit_id = np.zeros(1, dtype=np.int32)
    for tile in range(NUM_LEVELS):
        for points in (spawn[tile*count:tile*count+64], goals[tile*count:tile*count+64]):
            for point in points.numpy():
                origin = np.array([point[0], point[1], 5.], dtype=np.float64)
                length = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), None, True, -1, hit_id)
                assert length >= 0
                worst = max(worst, abs(5.-length))
                ray_count += 1
    assert worst < 1e-6
    return dict(sampled_pairs=n, distance_min=float(distance.min()), distance_max=float(distance.max()),
                spawn_yaw_min=float(face.min()), spawn_yaw_max=float(face.max()),
                goal_yaw_min=float(headings.min()), goal_yaw_max=float(headings.max()),
                ray_count=ray_count, max_ray_height_error=worst, seeded_replay=True)


def check_live(cfg):
    from instinct_mj.rl import InstinctRlVecEnvWrapper
    cfg.scene.num_envs = 4
    cfg.disable_randomization()
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.delay_max_lag = 0
    env = load_env_cls(TASK)(cfg=cfg, device="cuda:0")
    try:
        vec = InstinctRlVecEnvWrapper(env, policy_group="policy", critic_group="critic")
        obs, extras = vec.reset()
        for _ in range(3):
            planner = torch.cat((obs[:, :375], extras["observations"]["planner"], obs[:, 465:468]), dim=-1)
            assert planner.shape == (4, 939) and torch.isfinite(planner).all()
            obs, reward, _, extras = vec.step(torch.zeros(4, 23, device=env.device))
            assert torch.isfinite(reward).all()
        for term in ("move2goal", "standatgoal"):
            assert env.reward_manager.get_term_cfg(term).params["radius"] == .2
        assert env.reward_manager.get_term_cfg("standatgoal").params["heading_tolerance"] == .25
        return dict(num_envs=4, steps=3, planner_obs_dim=939, action_dim=23, rewards_finite=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-folder")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.motion_folder:
        buffer = g1.motion_reference_cfg.motion_buffers["TerrainMotion"]
        buffer.path = os.path.expanduser(args.motion_folder)
        buffer.metadata_yaml = os.path.join(buffer.path, "metadata.yaml")
    cfg = check_config()
    result = dict(task=TASK, config_isolated=True, sampling=check_sampling_and_rays(cfg))
    if args.live:
        result["live"] = check_live(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
