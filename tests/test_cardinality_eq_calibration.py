import unittest

import numpy as np
import pandas as pd

from cardinality_eq_calibration import (
    apply_equality_residual_calibrator,
    equality_calibration_keys,
    fit_equality_residual_calibrator,
)


class EqualityResidualCalibrationTest(unittest.TestCase):
    def test_equality_keys_ignore_non_equality_predicates(self) -> None:
        row = {
            "Tables": "cast_info ci",
            "Join Conditions": "",
            "Predicates": "ci.person_id,<,10",
        }

        self.assertEqual(equality_calibration_keys(row), {})

    def test_fit_and_apply_equality_residual_shift(self) -> None:
        train_df = pd.DataFrame(
            {
                "Tables": ["cast_info ci"] * 10,
                "Join Conditions": [""] * 10,
                "Predicates": ["ci.person_id,=,1"] * 10,
                "Cardinality": [100] * 10,
            }
        )
        true_log_y = np.full(10, 3.0)
        oof_log_pred = np.full(10, 2.0)
        calibrator = fit_equality_residual_calibrator(
            train_df,
            true_log_y,
            oof_log_pred,
            min_count=8,
        )
        target_df = pd.DataFrame(
            {
                "Tables": ["cast_info ci", "cast_info ci"],
                "Join Conditions": ["", ""],
                "Predicates": ["ci.person_id,=,1", "ci.person_id,<,1"],
            }
        )

        calibrated, metrics = apply_equality_residual_calibrator(
            target_df,
            np.array([2.0, 2.0]),
            calibrator,
        )

        self.assertGreater(calibrated[0], 2.0)
        self.assertEqual(calibrated[1], 2.0)
        self.assertEqual(metrics["eq_residual_calibration_coverage"], 0.5)

    def test_small_groups_are_not_used(self) -> None:
        train_df = pd.DataFrame(
            {
                "Tables": ["movie_companies mc"] * 2,
                "Join Conditions": [""] * 2,
                "Predicates": ["mc.company_id,=,1"] * 2,
                "Cardinality": [10] * 2,
            }
        )
        calibrator = fit_equality_residual_calibrator(
            train_df,
            np.array([5.0, 5.0]),
            np.array([1.0, 1.0]),
            min_count=8,
        )

        calibrated, metrics = apply_equality_residual_calibrator(
            train_df,
            np.array([1.0, 1.0]),
            calibrator,
        )

        np.testing.assert_allclose(calibrated, np.array([1.0, 1.0]))
        self.assertEqual(metrics["eq_residual_calibration_group_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
