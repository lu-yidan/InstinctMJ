"""Supervision audit rejects censored episodes and masks invalid positions."""
import unittest

import numpy as np

from audit_contact_labels import analyze, compare_targets, geometry_pairs, match_onsets, segments


class ContactLabelAuditTests(unittest.TestCase):
    def test_geometry_uses_estimated_frame_and_separates_link_from_surface_error(self):
        rows = dict(ref_valid=np.ones((5, 9), bool), ref_targets_local=np.zeros((5, 9, 3)),
                    training_estimated_pelvis_yaw=np.full(5, np.pi/2),
                    training_estimated_pelvis_pos_w=np.tile([5., 6., 1.], (5, 1)))
        for key in ('training_contact_pos_w', 'training_contact_link_pos_w', 'reference_contact_link_pos_w'):
            rows[key] = np.zeros((5, 9, 3))
        for key in ('training_contact_link_rot_w', 'reference_contact_link_rot_w'):
            rows[key] = np.tile(np.eye(3), (5, 9, 1, 1))
        rows['ref_targets_local'][1, 3] = [1., 0., -1.]
        rows['reference_contact_link_pos_w'][2, 3] = [5., 6.9, .04]
        rows['training_contact_link_pos_w'][3, 3] = [5.02, 6.92, .04]
        rows['training_contact_pos_w'][3, 3] = [5.03, 7.12, 0.]
        result = geometry_pairs(rows, np.arange(5), 3,
            [dict(start=2, left_censored=False)], [dict(start=3, length=10, left_censored=False)], 10)[0]
        np.testing.assert_allclose(result['contact_delta_yaw_m'], [.12, -.03, 0.], atol=1e-12)
        np.testing.assert_allclose(result['link_tracking_delta_yaw_m'], [.02, -.02, 0.], atol=1e-12)
        np.testing.assert_allclose(result['surface_offset_delta_yaw_m'], [.10, -.01, 0.], atol=1e-12)

    def test_onset_matching_is_one_to_one_and_respects_tolerance(self):
        self.assertEqual(match_onsets([10, 30, 60], [9, 11, 35, 90]), [(10, 9), (30, 35)])

    def test_segment_edges_retain_censoring(self):
        found = segments([1, 1, 0, 1, 0, 1, 1])
        self.assertEqual([(x['start'], x['end']) for x in found], [(0, 2), (3, 4), (5, 7)])
        self.assertTrue(found[0]['left_censored'])
        self.assertTrue(found[-1]['right_censored'])
        self.assertFalse(found[1]['left_censored'] or found[1]['right_censored'])

    def test_invalid_zero_vectors_are_not_position_errors(self):
        pos = np.array([[[1., 0., 0.], [200., 200., 200.]]])
        result = compare_targets(pos, [[True, True]], np.zeros((1, 2, 3)), [[True, False]], [True])
        self.assertEqual(result['both_valid'], 1)
        self.assertEqual(result['hindsight_only_valid'], 1)
        self.assertEqual(result['xy_error_median_m'], 1.)
        self.assertEqual(result['signed_xyz_mean_m'], [1., 0., 0.])

    def test_no_valid_target_has_no_position_statistic(self):
        result = compare_targets(np.zeros((2, 9, 3)), np.zeros((2, 9), bool),
                                 np.zeros((2, 9, 3)), np.zeros((2, 9), bool), [True, True])
        self.assertIsNone(result['xy_error_p90_m'])
        self.assertIsNone(result['signed_xyz_mean_m'])
        self.assertEqual(result['both_invalid'], 18)

    def test_missing_start_or_missing_end_cannot_silently_create_future_labels(self):
        base = dict(variant=np.ones(2, int), clip_index=np.zeros(2, int), episode=np.zeros(2, int),
                    tick=np.arange(2), done_after_step=np.array([False, False]))
        with self.assertRaisesRegex(ValueError, 'incomplete episode'):
            analyze(base, {})
        base['tick'] = np.array([1, 2])
        base['done_after_step'][-1] = True
        with self.assertRaisesRegex(ValueError, 'contiguous episode'):
            analyze(base, {})


if __name__ == '__main__':
    unittest.main()
