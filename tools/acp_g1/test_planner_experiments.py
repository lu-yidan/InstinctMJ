"""Stage 2 ablations retain optimizer history and do not alter default schedules."""
import ast
import copy
import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from run_stages import inherit_agent_config, override_planner_experiment, override_reward_handover_start, verify_frozen_planner

# Runner imports initialize Warp. Exercise its actual lightweight methods and
# the pure Torch trainer without creating a CUDA context or simulator.
source = Path(__file__).resolve().parents[2] / "src/instinct_mj/rl/acp"
package = types.ModuleType("acp_planner_experiment_unit")
package.__path__ = [str(source)]
sys.modules.setdefault(package.__name__, package)
stage2 = importlib.import_module(package.__name__ + ".stage2")
PlannerTrainer = stage2.PlannerTrainer
runner_ast = ast.parse((source / "runner.py").read_text())
runner_class = next(n for n in runner_ast.body if isinstance(n, ast.ClassDef) and n.name == "ACPRunner")
methods = [n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name in {"eta", "planner_update"}]
namespace = {"eta_schedule": stage2.eta}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(source / "runner.py"), "exec"), namespace)
ACPRunner = SimpleNamespace(**{name: namespace[name] for name in ("eta", "planner_update")})


class PlannerExperimentTests(unittest.TestCase):
    def test_delayed_reward_preserves_observation_takeover_and_real_clock(self):
        agent = SimpleNamespace(contact_source="mixed", eta_obs_start_iter=1000,
            eta_rew_start_iter=3000, eta_ann_iters=4000, handover_hold_iteration=None)
        override_reward_handover_start(agent, None)
        self.assertEqual(agent.eta_rew_start_iter, 3000)
        override_reward_handover_start(agent, 100000)
        runner = SimpleNamespace(it=4000, cfg=vars(agent))
        self.assertEqual(ACPRunner.eta(runner), (.75, 0.))
        runner.it = 5000
        self.assertEqual(ACPRunner.eta(runner), (1., 0.))
        self.assertEqual(runner.it, 5000)
        self.assertEqual(agent.eta_obs_start_iter, 1000)

    def test_invalid_reward_start_is_rejected_without_config_mutation(self):
        agent = SimpleNamespace(contact_source="mixed", eta_obs_start_iter=1000, eta_rew_start_iter=3000)
        for value in (-1, 0, 1000, 4000.5, True):
            with self.assertRaises(ValueError):
                override_reward_handover_start(agent, value)
            self.assertEqual(agent.eta_rew_start_iter, 3000)

    def test_handover_hold_does_not_reset_training_clock_or_change_other_modes(self):
        runner = SimpleNamespace(it=5000, cfg=dict(contact_source="mixed", eta_obs_start_iter=1000,
            eta_rew_start_iter=3000, eta_ann_iters=4000, handover_hold_iteration=None))
        self.assertEqual(ACPRunner.eta(runner), (1.0, .5))
        runner.cfg["handover_hold_iteration"] = 4000
        self.assertEqual(ACPRunner.eta(runner), (.75, .25))
        self.assertEqual(runner.it, 5000)
        runner.it = 3500
        self.assertEqual(ACPRunner.eta(runner), (.625, .125))
        runner.cfg["contact_source"] = "planner"
        self.assertEqual(ACPRunner.eta(runner), (1.0, 1.0))
        runner.cfg["contact_source"] = "reference"
        self.assertEqual(ACPRunner.eta(runner), (0.0, 0.0))

    def test_controls_follow_resume_and_can_be_released_independently(self):
        agent = SimpleNamespace(lr=.0002, pi_lr_scale=1., planner_lr=.0001,
            freeze_planner=False, handover_hold_iteration=None)
        inherit_agent_config(agent, dict(planner_lr=.00005, freeze_planner=True, handover_hold_iteration=4000))
        override_planner_experiment(agent)
        self.assertTrue(agent.freeze_planner)
        self.assertEqual(agent.handover_hold_iteration, 4000)
        override_planner_experiment(agent, planner_lr=.000025, release_hold=True, freeze=False)
        self.assertEqual(agent.planner_lr, .000025)
        self.assertIsNone(agent.handover_hold_iteration)
        self.assertFalse(agent.freeze_planner)
        self.assertEqual((agent.lr, agent.pi_lr_scale), (.0002, 1.))
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                override_planner_experiment(agent, planner_lr=value)
        for value in (-1, 4.5):
            with self.assertRaises(ValueError):
                override_planner_experiment(agent, hold_at=value)
        with self.assertRaises(ValueError):
            override_planner_experiment(agent, hold_at=4000, release_hold=True)

    def test_planner_resume_keeps_new_lr_old_weights_moments_and_counter(self):
        first = PlannerTrainer(lr=.0001, planner=torch.nn.Linear(3, 2))
        first.planner(torch.ones(4, 3)).square().sum().backward()
        first.opt.step()
        first.num_updates = 123
        saved = copy.deepcopy(first.state_dict())
        second = PlannerTrainer(lr=.000025, planner=torch.nn.Linear(3, 2))
        second.load_state_dict(saved)
        self.assertEqual(second.num_updates, 123)
        self.assertTrue(all(g["lr"] == .000025 for g in second.opt.param_groups))
        self.assertEqual(saved["opt"]["param_groups"][0]["lr"], .0001)
        for key, value in saved["planner"].items():
            self.assertTrue(torch.equal(value, second.planner.state_dict()[key]))
        for parameter, state in saved["opt"]["state"].items():
            for key, value in state.items():
                actual = second.opt.state_dict()["state"][parameter][key]
                self.assertTrue(torch.equal(value, actual) if torch.is_tensor(value) else value == actual)

    def test_frozen_planner_never_calls_optimizer_but_normal_updates_remain(self):
        trainer, planner = Mock(), Mock()
        trainer.update.return_value = {"loss": 1.}
        runner = SimpleNamespace(cfg={"freeze_planner": True}, planner_trainer=trainer, planner=planner, replay=object())
        self.assertIsNone(ACPRunner.planner_update(runner))
        trainer.update.assert_not_called()
        runner.cfg["freeze_planner"] = False
        self.assertEqual(ACPRunner.planner_update(runner), {"loss": 1.})
        trainer.update.assert_called_once_with(runner.replay)

    def test_frozen_checkpoint_verification_rejects_weight_or_dtype_drift(self):
        before = {"planner_state_dict": {"weight": torch.ones(2, 3)}}
        self.assertEqual(verify_frozen_planner(before, copy.deepcopy(before)), 1)
        after = copy.deepcopy(before)
        after["planner_state_dict"]["weight"][0, 0] += .01
        with self.assertRaisesRegex(ValueError, "weight"):
            verify_frozen_planner(before, after)


if __name__ == "__main__":
    unittest.main()
