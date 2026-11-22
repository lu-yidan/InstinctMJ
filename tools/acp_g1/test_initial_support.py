"""Initial support labeling regressions without constructing a simulator."""
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch

# Load the pure Torch modules without importing the simulator package.
package = types.ModuleType("acp_support_unit")
package.__path__ = [str(Path(__file__).resolve().parents[2]/"src/instinct_mj/rl/acp")]
sys.modules.setdefault(package.__name__, package)
hindsight = importlib.import_module(package.__name__+".hindsight")
stage2 = importlib.import_module(package.__name__+".stage2")


class InitialSupportTests(unittest.TestCase):
    def setUp(self):
        self.T = 50
        self.force = torch.zeros(self.T, 9)
        self.force[2:14, 3] = 10
        self.force[26:38, 3] = 10
        self.force[7:19, 4] = 10
        self.force[32:44, 4] = 10
        self.pos = torch.zeros(self.T, 9, 3)
        self.pos[:26, 3, 0] = -.17
        self.pos[26:, 3, 0] = .67
        self.pos[:, 4, 0] = .18
        self.base = torch.zeros(self.T, 3)
        self.yaw = torch.zeros(self.T)
        self.ref = torch.zeros(self.T, 9, dtype=torch.bool)
        self.ref[:6, 3] = True
        self.ref[26:38, 3] = True
        self.ref[7:19, 4] = True

    def labels(self, reference=None):
        return hindsight.label_episode(self.force, self.pos, self.base, self.yaw, reference_contacts=reference)

    def test_settling_support_is_skipped_but_initial_swing_landing_is_kept(self):
        old, _ = self.labels()
        new, valid = self.labels(self.ref)
        self.assertAlmostEqual(old[0, 3, 0].item(), -.17)
        self.assertAlmostEqual(new[0, 3, 0].item(), .67)
        self.assertAlmostEqual(new[0, 4, 0].item(), .18)
        self.assertTrue(valid[0, 3] and valid[0, 4])

    def test_contact_after_reference_liftoff_is_a_genuine_new_contact(self):
        self.ref[1:6, 3] = False
        old, old_valid = self.labels()
        new, new_valid = self.labels(self.ref)
        torch.testing.assert_close(old, new)
        self.assertTrue(torch.equal(old_valid, new_valid))

    def test_already_initialized_contact_keeps_all_original_labels(self):
        self.force[:2, 3] = 10
        old, old_valid = self.labels()
        new, new_valid = self.labels(self.ref)
        torch.testing.assert_close(old, new)
        self.assertTrue(torch.equal(old_valid, new_valid))

    def test_goal_episode_does_not_use_reference_support(self):
        kwargs = dict(proprio=torch.zeros(self.T, 375), goal=torch.zeros(self.T, 3),
                      points=torch.zeros(self.T, 187, 3), contact_force_z=self.force,
                      body_pos_w=self.pos, pelvis_pos_w=self.base, pelvis_yaw=self.yaw,
                      start_phase=0., end_phase=1., min_xy_ok=.1, near_idx=-1,
                      init_xy=1., work=1., terrain_type=0, tile=0, num_steps=self.T,
                      reference_contacts=self.ref)
        ref = stage2.label_episode_samples(stage2.Episode(is_goal=False, **kwargs))
        goal = stage2.label_episode_samples(stage2.Episode(is_goal=True, **kwargs))
        self.assertAlmostEqual(ref['target_pos'][0, 3, 0].item(), .67)
        self.assertAlmostEqual(goal['target_pos'][0, 3, 0].item(), -.17)

    def test_recorder_reference_flags_survive_partial_reset_without_aliasing(self):
        recorder = stage2.EpisodeRecorder(2, max_steps=6)
        kwargs = dict(proprio=torch.zeros(2, 375), goal=torch.zeros(2, 3), points=torch.zeros(2, 187, 3),
                      contact_force_z=torch.zeros(2, 9), body_pos_w=torch.zeros(2, 9, 3),
                      pelvis_pos_w=torch.zeros(2, 3), pelvis_yaw=torch.zeros(2),
                      is_goal=torch.zeros(2, dtype=torch.bool), phase=torch.zeros(2),
                      goal_xy=torch.ones(2), goal_dz=torch.zeros(2))
        for _ in range(2):
            recorder.step(**kwargs, reference_contacts=torch.ones(2, 9, dtype=torch.bool))
        first = recorder.finish(torch.tensor([0]), only_successful=False)[0]
        recorder.step(**kwargs, reference_contacts=torch.zeros(2, 9, dtype=torch.bool))
        second = recorder.finish(torch.tensor([0]), only_successful=False)[0]
        self.assertTrue(first.reference_contacts.all())
        self.assertFalse(second.reference_contacts.any())
        self.assertEqual(int(recorder.count[1]), 3)


if __name__ == '__main__':
    unittest.main()
