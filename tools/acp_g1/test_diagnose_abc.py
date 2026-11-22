"""Numerical/protocol regressions; these do not instantiate training or MuJoCo."""
import unittest

import numpy as np

from diagnose_abc import actor_reference, compare_initial_states, first_divergence, local_targets_to_world, signed_target_error, tensor_storage


class DiagnosticProtocolTests(unittest.TestCase):
    def test_reference_bypass_does_not_enter_planner_policy(self):
        class Actor:
            def act_inference(self, value):
                return value + 7
        class Runner:
            ac = Actor()
            normalizers = {"policy": lambda value: value * 2}
            def get_inference_policy(self, *args, **kwargs):
                raise AssertionError("Reference bypass entered planner policy")
            def plan_contacts(self, *args, **kwargs):
                raise AssertionError("Reference bypass replaced reference targets")
        np.testing.assert_array_equal(actor_reference(Runner(), np.array([1., 2.])), [9., 11.])

    def test_pairing_checks_world_goal_and_phase_not_only_physics(self):
        state = {"qpos": np.zeros((2, 30)), "qvel": np.zeros((2, 29)), "world_goal": np.zeros((2, 5)), "phase": np.zeros(2)}
        compare_initial_states(state, {key: value.copy() for key, value in state.items()})
        for field in ("world_goal", "phase"):
            changed = {key: value.copy() for key, value in state.items()}
            changed[field].flat[0] = .1
            with self.assertRaisesRegex(ValueError, field):
                compare_initial_states(state, changed)
        shared_nan = {"qvel": np.array([[np.nan]])}
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            compare_initial_states(shared_nan, shared_nan)

    def test_warp_proxy_state_copy_reaches_tensor_storage(self):
        class Proxy:
            def __init__(self, value):
                self._tensor = value
        backing = np.zeros((2, 3))
        proxy = Proxy(backing)
        tensor_storage(proxy)[:] = 2
        np.testing.assert_array_equal(backing, np.full((2, 3), 2))
        self.assertIs(tensor_storage(backing), backing)

    def test_yaw_frame_and_invalid_contact_error(self):
        target = np.array([[[1., 0., -.5], [0., 1., -.5]]])
        world = local_targets_to_world(target, [[True, False]], [[2., 3., 1.]], [np.pi / 2])
        np.testing.assert_allclose(world[0, 0], [2., 4., .5], atol=1e-12)
        self.assertTrue(np.isnan(world[0, 1]).all())
        delta, both = signed_target_error(target, [[True, True]], target + [.2, -.3, .1], [[True, False]])
        np.testing.assert_allclose(delta[0, 0], [.2, -.3, .1])
        self.assertTrue(np.isnan(delta[0, 1]).all())
        np.testing.assert_array_equal(both, [[True, False]])

    def test_numerical_action_difference_is_not_clear_gait_divergence(self):
        n = 8
        variants = np.repeat(np.arange(3), n)
        ticks = np.tile(np.arange(n), 3)
        positions = np.zeros((3 * n, 3))
        positions[2 * n + 2:2 * n + 7, 0] = .06
        actions = np.zeros((3 * n, 23))
        actions[2 * n:, 0] = .01
        contacts = np.zeros((3 * n, 9))
        for variant in range(3):
            contacts[variant * n:(variant + 1) * n, 3] = [0, 1, 1, 0, 0, 1, 1, 0]
        rows = {"variant": variants, "tick": ticks, "clip_index": np.zeros(3 * n, int), "episode": np.zeros(3 * n, int),
                "actions": actions, "pelvis_pos_w": positions, "measured_contact": contacts,
                "reference_contact_flags": contacts.copy(), "world_goal": np.zeros((3 * n, 5)),
                "target_error_xyz": np.zeros((3 * n, 9, 3))}
        result = first_divergence(rows)[0]
        self.assertEqual(result["first_action_difference_tick"], 0)
        self.assertEqual(result["first_clear_pelvis_xy_difference_gt_5cm_sustained_5ticks"], 2)
        cycle = result["measured_contact_cycle_at_clear_divergence"]["C_left"]
        self.assertEqual((cycle["cycle_start_tick"], cycle["cycle_end_tick"]), (1, 5))
        self.assertTrue(cycle["complete_onset_to_onset"])

    def test_policy_rng_fork_cannot_move_environment_stream(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is required for the RNG regression")
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.rand(4096)  # eta=1's irrelevant Bernoulli draws
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        actual = torch.rand(8)
        torch.set_rng_state(before)
        self.assertTrue(torch.equal(actual, torch.rand(8)))


if __name__ == "__main__":
    unittest.main()
