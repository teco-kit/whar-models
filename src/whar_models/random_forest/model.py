from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.classical import ClassicalHARModel


class RandomForestHAR(ClassicalHARModel):
    NAME = "Random Forest"
    display_name = "Random Forest"
    color = "#2a9d8f"
    ARCHITECTURE = "tsfresh MinimalFCParameters + RandomForestClassifier"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(feature_engineering=True, classical_ml=True)
    NOTES = "Classical baseline with tsfresh minimal feature set and a random forest."

    def _build_estimator(self, **estimator_kwargs: Any) -> RandomForestClassifier:
        defaults: dict[str, Any] = {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
            "max_depth": 8,
            "min_samples_leaf": 10,
        }
        defaults.update(estimator_kwargs)
        return RandomForestClassifier(**defaults)


def build_random_forest(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
) -> RandomForestHAR:
    return RandomForestHAR(
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
