"""Frozen tracker regressions; no simulator or training process is created."""
import copy
from types import SimpleNamespace
import unittest

import torch

from run_stages import inherit_agent_config, verify_frozen_tracker


class PlannerRefitTests(unittest.TestCase):
    def setUp(self):
        self.saved = {
            "model_state_dict": {"actor.weight": torch.tensor([[1., 2.]])},
            "policy_normalizer_state_dict": {"mean": torch.zeros(2), "count": torch.tensor(123.)},
        }

    def test_unchanged_tracker_passes_but_actor_or_normalizer_drift_fails(self):
        self.assertEqual(verify_frozen_tracker(self.saved, copy.deepcopy(self.saved)), 3)
        for group, name in [("model_state_dict", "actor.weight"), ("policy_normalizer_state_dict", "count")]:
            with self.subTest(group=group):
                changed = copy.deepcopy(self.saved)
                changed[group][name] += 1
                with self.assertRaisesRegex(ValueError, name):
                    verify_frozen_tracker(self.saved, changed)

    def test_changed_keys_or_dtype_are_not_a_verified_freeze(self):
        changed = copy.deepcopy(self.saved)
        changed["model_state_dict"]["new_buffer"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "keys"):
            verify_frozen_tracker(self.saved, changed)
        changed = copy.deepcopy(self.saved)
        changed["policy_normalizer_state_dict"]["mean"] = torch.zeros(2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "mean"):
            verify_frozen_tracker(self.saved, changed)

    def test_freeze_is_explicit_for_each_new_run(self):
        target = SimpleNamespace(freeze_tracker=False, lr=.1)
        inherit_agent_config(target, {"freeze_tracker": True, "lr": .2})
        self.assertFalse(target.freeze_tracker)
        self.assertEqual(target.lr, .2)


if __name__ == "__main__":
    unittest.main()
