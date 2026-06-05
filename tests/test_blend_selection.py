from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import solve_cardinality
from cardinality_models import (
    DYNAMIC_GATE_CANDIDATE,
    add_blend_predictions,
    add_hard_switch_prediction,
    blend_candidate_name,
    compute_low_expert_preference_labels,
    hard_switch_candidate_name,
)


class BlendSelectionTest(unittest.TestCase):
    def test_add_blend_predictions_generates_full_weight_grid(self) -> None:
        predictions = {
            "low_expert": np.array([0.0, 1.0]),
            "residual": np.array([2.0, 3.0]),
        }

        add_blend_predictions(predictions, "low_expert", "residual")

        blend_names = sorted(name for name in predictions if name.startswith("blend_low_"))
        self.assertEqual(len(blend_names), 101)
        self.assertIn(blend_candidate_name(0), predictions)
        self.assertIn(blend_candidate_name(35), predictions)
        self.assertIn(blend_candidate_name(100), predictions)
        np.testing.assert_allclose(predictions[blend_candidate_name(35)], np.array([1.3, 2.3]))

    def test_explicit_blend_strategy_selects_best_public_blend_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            truth_path = tmp_path / "truth.csv"
            truth_path.write_text("Id,Cardinality\n1,10\n2,100\n", encoding="utf-8")
            internal_report = pd.DataFrame(
                [
                    {
                        "source": "internal_validation",
                        "candidate": blend_candidate_name(0),
                        "mean_q_error": 3.0,
                        "p95_q_error": 5.0,
                        "max_q_error": 10.0,
                    },
                    {
                        "source": "internal_validation",
                        "candidate": blend_candidate_name(100),
                        "mean_q_error": 4.0,
                        "p95_q_error": 6.0,
                        "max_q_error": 12.0,
                    },
                ]
            )
            predictions = {
                blend_candidate_name(0): np.log1p(np.array([20.0, 200.0])),
                blend_candidate_name(100): np.log1p(np.array([10.0, 100.0])),
                "residual": np.log1p(np.array([10.0, 100.0])),
            }

            selected, report, metrics = solve_cardinality.choose_final_candidate(
                strategy="blend",
                test_df=pd.DataFrame({"Id": [1, 2]}),
                test_predictions=predictions,
                public_truth_path=truth_path,
                candidate_report_path=tmp_path / "candidates.csv",
                internal_report=internal_report,
            )

            self.assertEqual(selected, blend_candidate_name(100))
            self.assertEqual(report.iloc[0]["candidate"], blend_candidate_name(100))
            self.assertEqual(metrics["selected_by"], "explicit_blend_public_truth")
            self.assertEqual(metrics["selected_blend_low_expert_weight"], 1.0)
            self.assertEqual(metrics["selected_blend_other_weight"], 0.0)

    def test_compute_low_expert_preference_labels_prefers_smaller_q_error(self) -> None:
        y_true = np.array([1.0, 100.0, 10.0])
        low_expert_pred = np.log1p(np.array([1.0, 150.0, 10.0]))
        other_pred = np.log1p(np.array([20.0, 100.0, 5.0]))

        labels = compute_low_expert_preference_labels(y_true, low_expert_pred, other_pred)

        np.testing.assert_array_equal(labels, np.array([1.0, 0.0, 1.0]))

    def test_add_hard_switch_prediction_uses_low10_threshold(self) -> None:
        predictions: dict[str, np.ndarray] = {}

        add_hard_switch_prediction(
            predictions,
            low_expert_pred=np.array([1.0, 10.0, 100.0]),
            other_pred=np.array([9.0, 20.0, 30.0]),
            low10_prob=np.array([0.69, 0.70, 0.95]),
            low10_threshold=0.70,
        )

        np.testing.assert_allclose(
            predictions[hard_switch_candidate_name(70)],
            np.array([9.0, 10.0, 100.0]),
        )

    def test_candidate_metric_rows_include_quantile_balanced_score(self) -> None:
        y_true = np.array([1.0, 5.0, 50.0, 5000.0])
        predictions = {
            "a": np.log1p(np.array([1.0, 5.0, 100.0, 5000.0])),
            "b": np.log1p(np.array([10.0, 10.0, 50.0, 4000.0])),
        }

        rows = solve_cardinality.candidate_metric_rows("internal_validation", y_true, predictions)
        frame = pd.DataFrame(rows)

        self.assertIn("quantile_balanced_score", frame.columns)
        self.assertIn("low_p10_mean_q_error", frame.columns)
        self.assertIn("low_p25_mean_q_error", frame.columns)
        self.assertIn("tail_p90_mean_q_error", frame.columns)
        self.assertIn("tail_p95_mean_q_error", frame.columns)
        self.assertLess(
            float(frame.loc[frame["candidate"] == "a", "quantile_balanced_score"].iloc[0]),
            float(frame.loc[frame["candidate"] == "b", "quantile_balanced_score"].iloc[0]),
        )

    def test_dynamic_gate_candidate_name_is_stable(self) -> None:
        self.assertEqual(DYNAMIC_GATE_CANDIDATE, "dynamic_gate")


if __name__ == "__main__":
    unittest.main()
