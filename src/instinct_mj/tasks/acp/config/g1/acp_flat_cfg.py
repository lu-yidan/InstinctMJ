"""Deployment-focused Stage 3 on flat ground; the paper's 17-terrain task is unchanged.

The training task keeps Stage 3's mixed reference/goal batches, rewards and
runner. Its goal terrain alone is flat, and its starting yaw is uniform over
[-pi, pi] rather than the original +/-0.4 rad. This intentional custom curriculum
teaches turning toward side/back targets and stopping at arbitrary target yaw.
Its move/stand reward radius is 0.2 m and stand heading tolerance is 0.25 rad,
matching the deployment baseline; the goal-command scaling radius stays 0.5 m.
The analytic Stage 3 MDP reads these reward parameters from the live manager.
Play retains seeded goal/reset sampling while disabling the original domain
randomization, observation corruption and action delay.
"""
import math
from dataclasses import dataclass, field

import instinct_mj.tasks.acp.acp_env_cfg as acp_cfg
from instinct_mj.tasks.acp.acp_stage3_env_cfg import AcpStage3SceneCfg
from instinct_mj.tasks.acp.goal_terrains import GoalArenaCfg

from .acp_stage3_cfg import G1AcpStage3V1EnvCfg, G1AcpStage3V1EnvCfg_PLAY
from .acp_tracking_cfg import make_scene_cfg


def _configure_flat_goal(env_cfg):
    env_cfg.commands[acp_cfg.GOAL_COMMAND].reset_yaw_range = math.pi
    rewards = env_cfg.rewards["rewards"]
    rewards["move2goal"].params["radius"] = 0.2
    rewards["standatgoal"].params.update(radius=0.2, heading_tolerance=0.25)


@dataclass(kw_only=True)
class AcpStage3FlatSceneCfg(AcpStage3SceneCfg):
    # Supply the type before AcpStage3SceneCfg.__post_init__ builds self.arena;
    # changing arena_cfg.types afterwards would leave stale geometry/tables.
    arena_cfg: GoalArenaCfg = field(default_factory=lambda: GoalArenaCfg(types=("flat",)))


@dataclass(kw_only=True)
class G1AcpStage3FlatV1EnvCfg(G1AcpStage3V1EnvCfg):
    scene: AcpStage3FlatSceneCfg = field(default_factory=lambda: make_scene_cfg(AcpStage3FlatSceneCfg, num_envs=2048))
    run_name: str = "g1AcpStage3FlatV1"

    def __post_init__(self):
        super().__post_init__()
        _configure_flat_goal(self)


@dataclass(kw_only=True)
class G1AcpStage3FlatV1EnvCfg_PLAY(G1AcpStage3V1EnvCfg_PLAY):
    scene: AcpStage3FlatSceneCfg = field(
        default_factory=lambda: make_scene_cfg(AcpStage3FlatSceneCfg, num_envs=1, env_spacing=2.5, num_tiles=1)
    )
    run_name: str = "g1AcpStage3FlatV1_play"

    def __post_init__(self):
        super().__post_init__()
        _configure_flat_goal(self)
