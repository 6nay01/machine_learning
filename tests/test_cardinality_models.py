import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import cardinality_models as cardinality_models


class _FakeBooster:
    def predict(self, dmatrix, iteration_range=None):
        return np.zeros(dmatrix.num_row(), dtype=float)


class CardinalityModelsTest(unittest.TestCase):
    def test_fit_main_model_without_early_stopping_uses_last_round(self) -> None:
        train_features = pd.DataFrame({"f1": [0.0, 1.0, 2.0]})
        target_features = pd.DataFrame({"f1": [3.0, 4.0]})
        y_train_log = np.array([0.0, 1.0, 2.0])
        y_target_log = np.array([3.0, 4.0])
        config = cardinality_models.XGBRegressorConfig(
            name="no_early_stop",
            params=cardinality_models.make_xgb_params(max_depth=2),
            num_boost_round=17,
            early_stopping_rounds=0,
            verbose_eval=0,
        )

        def fake_train(*, params, dtrain, num_boost_round, evals, early_stopping_rounds, verbose_eval):
            self.assertIsNone(early_stopping_rounds)
            self.assertEqual(num_boost_round, 17)
            return _FakeBooster()

        with patch.object(cardinality_models.xgb, "train", side_effect=fake_train):
            result = cardinality_models.fit_main_model(
                config=config,
                train_features=train_features,
                target_features=target_features,
                y_train_log=y_train_log,
                y_target_log=y_target_log,
            )

        self.assertEqual(result.best_iteration, 16)
        np.testing.assert_allclose(result.target_log_pred, np.zeros(len(target_features), dtype=float))

    def test_fit_oof_base_predictions_does_not_cap_rounds_at_900(self) -> None:
        train_features = pd.DataFrame({"f1": np.linspace(0.0, 1.0, num=6)})
        y_train_log = np.linspace(0.0, 1.0, num=6)
        config = cardinality_models.default_residual_config(max_rounds=700)
        observed_rounds: list[int] = []

        def fake_train(*, params, dtrain, num_boost_round, evals, verbose_eval):
            observed_rounds.append(int(num_boost_round))
            return _FakeBooster()

        with patch.object(cardinality_models.xgb, "train", side_effect=fake_train):
            predictions = cardinality_models.fit_oof_base_predictions(
                config=config,
                train_features=train_features,
                y_train_log=y_train_log,
                num_rounds=3200,
                folds=3,
            )

        np.testing.assert_allclose(predictions, np.zeros(len(train_features), dtype=float))
        self.assertEqual(observed_rounds, [3200, 3200, 3200])


if __name__ == "__main__":
    unittest.main()
