from __future__ import annotations

from typing import Any

from sklearn.svm import SVC

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.classical import ClassicalHARModel


class SVMHAR(ClassicalHARModel):
    NAME = "SVM"
    display_name = "SVM"
    color = "#00b4d8"
    ARCHITECTURE = "tsfresh MinimalFCParameters + SVC"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(feature_engineering=True, classical_ml=True)
    NOTES = "Classical baseline with tsfresh minimal feature set and an RBF-kernel SVM."

    def _build_estimator(self, **estimator_kwargs: Any) -> SVC:
        defaults: dict[str, Any] = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": False,
        }
        defaults.update(estimator_kwargs)
        return SVC(**defaults)


def build_svm(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
) -> SVMHAR:
    return SVMHAR(
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
