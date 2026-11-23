"""Fixed-goal geometry, hold semantics and Stage3 budget regression tests."""
import unittest
import numpy as np

from bounded_stage3_train import check_transition
from diagnose_goal_cohort import fixed_targets, hold_metrics


class GoalCohortTests(unittest.TestCase):
    def test_targets_rotate_with_pelvis_yaw_and_stop_is_stationary(self):
        spawn = np.tile([10.,20.,.82], (6,1))
        yaw = np.full(6,np.pi/2)
        targets, heading, cases = fixed_targets(spawn,yaw,np.zeros(6))
        np.testing.assert_array_equal(cases,[0,1,2,0,1,2])
        np.testing.assert_allclose(targets[0],[10.,21.,.78],atol=1e-12)
        np.testing.assert_allclose(targets[1],[9.,20.,.78],atol=1e-12)
        np.testing.assert_allclose(targets[4],[11.5,20.,.78],atol=1e-12)
        np.testing.assert_allclose(targets[[2,5],:2],spawn[[2,5],:2])
        self.assertAlmostEqual(heading[1],np.pi)
        self.assertAlmostEqual(heading[4],0.)
        self.assertLessEqual(np.linalg.norm(targets[:,:2]-spawn[:,:2],axis=-1).max(),2.)

    def test_pass_through_is_not_final_hold_and_fall_invalidates_hold(self):
        standing = np.array([[1,0,1],[1,1,1],[0,1,1],[0,1,1]],bool)
        alive = np.ones_like(standing); alive[2:,2] = False
        ever,final = hold_metrics(standing,alive,2)
        np.testing.assert_array_equal(ever,[True,True,True])
        np.testing.assert_array_equal(final,[False,True,False])

    def test_rejects_inherited_experiment_flags_or_wrong_clock(self):
        cfg=dict(contact_source="planner",train_planner=True,restart_iteration=True,
            freeze_tracker=False,freeze_planner=False,handover_hold_iteration=None,
            wm_only_iters=0,vfs_warmup_iters=500,planner_seed_initial_support=False,
            max_iterations=2000,save_interval=1000)
        check_transition(cfg,0,9000)
        for key,value in (("freeze_tracker",True),("freeze_planner",True),("wm_only_iters",200),
                          ("handover_hold_iteration",4000),("max_iterations",2001),("vfs_warmup_iters",0)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                check_transition(dict(cfg,**{key:value}),0,9000)
        with self.assertRaises(ValueError):check_transition(cfg,9000,9000)


if __name__ == "__main__":
    unittest.main()
