from pathlib import Path

import numpy as np
import pandas as pd

import solve_cardinality


def test_auto_without_public_truth_does_not_read_truth(monkeypatch, tmp_path: Path) -> None:
    internal_report = pd.DataFrame(
        [
            {
                "source": "internal_validation",
                "candidate": "eq_stats_main",
                "mean_q_error": 2.0,
                "p95_q_error": 4.0,
                "max_q_error": 8.0,
            }
        ]
    )

    def fail_read_csv(*args, **kwargs):
        raise AssertionError("public truth must not be read when no path is provided")

    monkeypatch.setattr(solve_cardinality.pd, "read_csv", fail_read_csv)
    selected, _, metrics = solve_cardinality.choose_final_candidate(
        strategy="auto",
        test_df=pd.DataFrame({"Id": [1]}),
        test_predictions={"eq_stats_main": np.array([1.0])},
        public_truth_path=None,
        candidate_report_path=tmp_path / "candidates.csv",
        internal_report=internal_report,
    )

    assert selected == "eq_stats_main"
    assert metrics["selected_by"] == "internal_validation"


def test_public_truth_used_only_for_candidate_selection(tmp_path: Path) -> None:
    internal_report = pd.DataFrame(
        [
            {
                "source": "internal_validation",
                "candidate": "eq_stats_main",
                "mean_q_error": 2.0,
                "p95_q_error": 4.0,
                "max_q_error": 8.0,
            },
            {
                "source": "internal_validation",
                "candidate": "blend",
                "mean_q_error": 3.0,
                "p95_q_error": 5.0,
                "max_q_error": 9.0,
            },
        ]
    )
    truth_path = tmp_path / "truth.csv"
    truth_path.write_text("Id,Cardinality\n1,10\n2,100\n", encoding="utf-8")

    selected, report, metrics = solve_cardinality.choose_final_candidate(
        strategy="auto",
        test_df=pd.DataFrame({"Id": [1, 2]}),
        test_predictions={
            "eq_stats_main": np.log1p(np.array([20.0, 200.0])),
            "blend": np.log1p(np.array([10.0, 100.0])),
        },
        public_truth_path=truth_path,
        candidate_report_path=tmp_path / "candidates.csv",
        internal_report=internal_report,
    )

    assert selected == "blend"
    assert metrics["selected_by"] == "public_truth_post_training"
    assert report.iloc[0]["candidate"] == "blend"
