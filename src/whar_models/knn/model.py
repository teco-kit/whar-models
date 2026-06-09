from __future__ import annotations

from typing import Any

from sklearn.neighbors import KNeighborsClassifier

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.classical import ClassicalHARModel


class KNNHAR(ClassicalHARModel):
    NAME = "K-NN"
    display_name = "k-NN"
    color = "#1d3557"
    ARCHITECTURE = "tsfresh MinimalFCParameters + KNeighborsClassifier"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(feature_engineering=True, classical_ml=True)
    NOTES = "Classical baseline with tsfresh minimal feature set and distance-based k-NN."

    def _build_estimator(self, **estimator_kwargs: Any) -> KNeighborsClassifier:
        defaults: dict[str, Any] = {
            "n_neighbors": 5,
            "weights": "distance",
        }
        defaults.update(estimator_kwargs)
        return KNeighborsClassifier(**defaults)


def build_knn(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
) -> KNNHAR:
    return KNNHAR(
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
