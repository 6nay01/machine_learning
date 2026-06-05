import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import grid_search_cardinality
from cardinality_models import add_blend_predictions, blend_candidate_name


class GridSearchCardinalityTest(unittest.TestCase):
    def test_add_blend_predictions_supports_custom_weight_steps(self) -> None:
        predictions = {
            "low_expert": np.array([0.0, 1.0]),
            "residual": np.array([2.0, 3.0]),
        }

        add_blend_predictions(
            predictions,
            "low_expert",
            "residual",
            blend_weight_steps=[0, 30, 100],
        )

        blend_names = sorted(name for name in predictions if name.startswith("blend_low_"))
        self.assertEqual(
            blend_names,
            [
                blend_candidate_name(0),
                blend_candidate_name(30),
                blend_candidate_name(100),
            ],
        )
        np.testing.assert_allclose(predictions[blend_candidate_name(30)], np.array([1.4, 2.4]))

    def test_iter_search_trials_applies_max_trials_limit(self) -> None:
        trials = grid_search_cardinality.iter_search_trials(
            {
                "a": [1, 2],
                "b": [10, 20],
                "c": [100],
            },
            max_trials=3,
        )

        self.assertEqual(len(trials), 3)
        self.assertEqual(trials[0], {"a": 1, "b": 10, "c": 100})
        self.assertEqual(trials[-1], {"a": 2, "b": 10, "c": 100})

    def test_evaluate_trial_selects_best_candidate_and_preserves_trial_metadata(self) -> None:
        original_candidate_predictions = grid_search_cardinality.candidate_predictions

        def fake_candidate_predictions(**kwargs):
            self.assertEqual(kwargs["blend_weight_steps"], [30])
            predictions = {
                "main_tuned": np.log1p(np.array([20.0, 200.0])),
                "low_expert": np.log1p(np.array([12.0, 120.0])),
                "residual": np.log1p(np.array([15.0, 150.0])),
                blend_candidate_name(30): np.log1p(np.array([10.0, 100.0])),
            }
            return predictions, {"selected_main_config": "main_tuned"}

        grid_search_cardinality.candidate_predictions = fake_candidate_predictions
        try:
            trial = {
                "main_eta": 0.03,
                "main_max_depth": 6,
                "main_min_child_weight": 1.5,
                "main_subsample": 0.92,
                "main_colsample_bytree": 0.92,
                "main_lambda": 1.2,
                "main_alpha": 0.02,
                "main_num_boost_round": 3200,
                "low_eta": 0.04,
                "low_max_depth": 4,
                "low_min_child_weight": 1.0,
                "low_subsample": 0.95,
                "low_colsample_bytree": 0.95,
                "low_lambda": 1.0,
                "low_alpha": 0.02,
                "low_num_boost_round": 1200,
                "residual_eta": 0.03,
                "residual_max_depth": 4,
                "residual_min_child_weight": 2.0,
                "residual_subsample": 0.90,
                "residual_colsample_bytree": 0.90,
                "residual_lambda": 3.0,
                "residual_alpha": 0.10,
                "residual_num_boost_round": 700,
                "blend_low_expert_weight_percent": 30,
            }

            result, candidate_report, payload = grid_search_cardinality.evaluate_trial(
                trial_id=7,
                trial=trial,
                train_df=pd.DataFrame({"Cardinality": [1, 2]}),
                train_features=pd.DataFrame({"f1": [0.0, 1.0]}),
                test_df=pd.DataFrame({"Id": [1, 2]}),
                test_features=pd.DataFrame({"f1": [0.0, 1.0]}),
                y_train=pd.Series([1.0, 2.0]),
                truth_df=pd.DataFrame({"Id": [1, 2], "Cardinality": [10, 100]}),
            )
        finally:
            grid_search_cardinality.candidate_predictions = original_candidate_predictions

        self.assertEqual(result["trial_id"], 7)
        self.assertEqual(result["selected_candidate"], blend_candidate_name(30))
        self.assertEqual(result["mean_q_error"], 1.0)
        self.assertEqual(result["selected_blend_low_expert_weight"], 0.3)
        self.assertEqual(result["selected_blend_other_weight"], 0.7)
        self.assertEqual(result["selected_main_config"], "main_tuned")
        self.assertFalse(candidate_report.empty)
        self.assertEqual(payload["selected_candidate"], blend_candidate_name(30))

    def test_sort_trial_results_uses_requested_metric_priority(self) -> None:
        results_df = pd.DataFrame(
            [
                {"trial_id": 1, "mean_q_error": 2.0, "p95_q_error": 4.0, "max_q_error": 9.0, "median_q_error": 1.5},
                {"trial_id": 2, "mean_q_error": 2.0, "p95_q_error": 3.0, "max_q_error": 10.0, "median_q_error": 1.2},
                {"trial_id": 3, "mean_q_error": 1.8, "p95_q_error": 5.0, "max_q_error": 12.0, "median_q_error": 1.8},
            ]
        )

        ranked = grid_search_cardinality.sort_trial_results(results_df)
        self.assertEqual(list(ranked["trial_id"]), [3, 2, 1])

    def test_trial_key_from_values_treats_int_and_float_consistently(self) -> None:
        key1 = grid_search_cardinality.trial_key_from_values(
            {"main_eta": 0.03, "blend_low_expert_weight_percent": 30},
            search_space_keys=("main_eta", "blend_low_expert_weight_percent"),
        )
        key2 = grid_search_cardinality.trial_key_from_values(
            {"main_eta": 0.030, "blend_low_expert_weight_percent": 30.0},
            search_space_keys=("main_eta", "blend_low_expert_weight_percent"),
        )
        self.assertEqual(key1, key2)

    def test_load_completed_trials_reads_existing_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trials_path = Path(tmpdir) / "grid_search_trials.csv"
            pd.DataFrame(
                [
                    {"trial_id": 1, "main_eta": 0.03, "blend_low_expert_weight_percent": 30, "mean_q_error": 2.0},
                    {"trial_id": 2, "main_eta": 0.035, "blend_low_expert_weight_percent": 40, "mean_q_error": 1.8},
                ]
            ).to_csv(trials_path, index=False)

            rows, completed = grid_search_cardinality.load_completed_trials(
                trials_path,
                search_space_keys=("main_eta", "blend_low_expert_weight_percent"),
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(len(completed), 2)
            self.assertIn(("0.03", "30"), completed)
            self.assertIn(("0.035", "40"), completed)

    def test_select_best_result_returns_lowest_metric_row(self) -> None:
        best = grid_search_cardinality.select_best_result(
            [
                {"trial_id": 1, "mean_q_error": 2.0, "p95_q_error": 4.0, "max_q_error": 9.0, "median_q_error": 1.5},
                {"trial_id": 2, "mean_q_error": 1.9, "p95_q_error": 5.0, "max_q_error": 10.0, "median_q_error": 1.2},
                {"trial_id": 3, "mean_q_error": 1.9, "p95_q_error": 4.5, "max_q_error": 11.0, "median_q_error": 1.1},
            ]
        )

        self.assertIsNotNone(best)
        self.assertEqual(best["trial_id"], 3)


if __name__ == "__main__":
    unittest.main()
