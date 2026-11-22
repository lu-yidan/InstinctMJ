#!/usr/bin/env python3
"""Native ACP Stage 3 flat reach/turn/stop evaluation, with no training updates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--task', default='Instinct-ACP-Stage3-Flat-G1-Play-v1')
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--min-success', type=float, default=.9)
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1 or not 0 <= args.min_success <= 1:
        parser.error('Invalid episode count or success threshold')
    checkpoint, dataset = args.checkpoint.expanduser().resolve(strict=True), args.dataset.expanduser().resolve(strict=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    import torch
    import yaml
    import instinct_mj.tasks  # noqa: F401
    from instinct_mj.tasks.acp.config.g1 import acp_tracking_cfg
    from instinct_mj.tasks.acp.goal_terrains import NUM_LEVELS
    from instinct_mj.scripts.instinct_rl.play import _disable_headless_debug_visualization
    from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg, load_runner_cls, load_vecenv_cls

    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not ck.get('has_planner') or not isinstance(ck.get('infos'),dict) or ck['infos'].get('stage') != 3:
        raise ValueError('Evaluation requires an actual Stage 3 checkpoint with planner')
    torch.manual_seed(args.seed)
    acp_tracking_cfg.MOTION_FOLDER = str(dataset)
    buffer = acp_tracking_cfg.motion_reference_cfg.motion_buffers['TerrainMotion']
    buffer.path, buffer.metadata_yaml = str(dataset), str(dataset/'metadata.yaml')
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = min(args.episodes, args.num_envs)
    cfg.seed, cfg.auto_reset = args.seed, False
    # Evaluation levels are a fixed cohort, not a continuation of training.
    # Otherwise reset() applies last episode's promote/demote after level is set.
    cfg.curriculum.pop('goal_terrain', None)
    _disable_headless_debug_visualization(cfg)
    if cfg.scene.arena.types != ('flat',):
        raise ValueError('This acceptance test requires the dedicated flat arena')
    agent = load_instinct_rl_cfg(args.task)
    agent_dict = agent.to_dict()
    saved = checkpoint.parent/'params/agent.yaml'
    if saved.is_file():
        values = yaml.safe_load(saved.read_text())
        agent_dict.update({key:value for key,value in values.items() if key in agent_dict and not isinstance(value,dict)})
    agent_dict.update(compile=False,compile_policy=False)
    raw = load_env_cls(args.task)(cfg=cfg,device='cuda:0')
    vec = load_vecenv_cls(args.task)(raw,policy_group=agent.policy_observation_group,
                                   critic_group=agent.critic_observation_group)
    started, results = time.monotonic(), []
    try:
        runner = load_runner_cls(args.task)(vec,agent_dict,log_dir=None,device='cuda:0')
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device='cuda:0')
        # setup_training() intentionally isn't needed: Play uses only goal envs.
        goal = raw.command_manager.get_term('acp_goal')
        reference = raw.command_manager.get_term('acp_reference_contacts')
        hold_frames = math.ceil(1./raw.step_dt)
        steps = math.ceil(goal.cfg.goal_episode_s/raw.step_dt)
        count, n = args.episodes, cfg.scene.num_envs
        with torch.inference_mode():
            for first in range(0,count,n):
                size = min(n,count-first)
                goal.level[:] = (torch.arange(n,device='cuda:0')+first) % NUM_LEVELS
                obs,_ = vec.reset()
                active = torch.arange(n,device='cuda:0') < size
                held = torch.zeros(n,dtype=torch.int64,device='cuda:0')
                completed = [None]*size
                initial = reference.base_pos_w.clone()
                targets = goal.target_pos_w.clone()
                headings = goal.target_yaw.clone()
                level = goal.level.clone()
                for tick in range(steps):
                    action = policy(obs)
                    finite_action = torch.isfinite(action).all(-1)
                    action = torch.where(finite_action[:,None],action,torch.zeros_like(action))
                    obs,_,done,_ = vec.step(action)
                    pos = reference.base_pos_w
                    velocity = reference.base_lin_vel_w
                    quat = reference.base_quat_w
                    psi = torch.atan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2))
                    heading_error = torch.atan2(torch.sin(headings-psi),torch.cos(headings-psi)).abs()
                    distance = (pos[:,:2]-targets[:,:2]).norm(dim=-1)
                    speed = velocity[:,:2].norm(dim=-1)
                    finite = torch.stack([torch.isfinite(value).flatten(1).all(-1)
                        for value in (pos,velocity,quat,obs)]).all(0) & finite_action
                    upright = 1-2*(quat[:,1]**2+quat[:,2]**2) > math.cos(.7)
                    standing = ((distance < .2)&(heading_error < .25)&(speed < .15)
                        &((pos[:,2]-targets[:,2]).abs() < .1)&upright&finite&~raw.reset_terminated)
                    held = torch.where(standing,held+1,torch.zeros_like(held))
                    success = held >= hold_frames
                    finish = active & (success|done.bool()|~finite)
                    if tick == steps-1: finish |= active
                    for index in finish.nonzero().flatten().tolist():
                        completed[index] = {'episode':first+index,'level':int(level[index]),
                            'success':bool(success[index]),'elapsed_s':(tick+1)*raw.step_dt,
                            'distance':float(distance[index]),'heading_error':float(heading_error[index]),
                            'speed_xy':float(speed[index]),'finite':bool(finite[index]),
                            'spawn':initial[index].cpu().tolist(),'target':targets[index].cpu().tolist(),
                            'target_heading':float(headings[index]),'final_position':pos[index].cpu().tolist()}
                    active &= ~finish
                    if not active.any(): break
                results.extend(completed)
        success = sum(row['success'] for row in results)/len(results)
        report = {'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'episode_count':len(results),
            'task':args.task,'dataset':str(dataset),'seed':args.seed,'gpu':args.gpu,
            'overall':{'success':success,'episodes':len(results),'finite':all(row['finite'] for row in results)},
            'criteria':{'distance':.2,'heading_error':.25,'speed_xy':.15,'height_error':.1,'max_tilt_rad':.7,'hold_seconds':1.,'minimum_success':args.min_success,
                        'success_semantics':'first continuous one-second hold at any time within the episode',
                        'goal_terrain_curriculum':False},
            'wall_seconds':time.monotonic()-started,'episodes':results,
            'passed':success >= args.min_success and all(row['finite'] for row in results)}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'overall':report['overall'],'passed':report['passed'],'output':str(args.output)}),flush=True)
    finally:
        vec.close()
    raise SystemExit(0 if report['passed'] else 2)


if __name__ == '__main__':
    main()
