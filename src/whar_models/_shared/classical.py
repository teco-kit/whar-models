from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn import __version__ as sklearn_version
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from whar_models._shared.architecture import ArchitectureSpec

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray | torch.Tensor


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str


class ClassicalHARModel(nn.Module, ABC):
    NAME: str | None = None
    display_name: str | None = None
    color: str | None = None
    ARCHITECTURE: str | None = None
    INPUT_TYPE: str = "TS"
    SOURCE: str = "scikit-learn + tsfresh"
    NOTES: str = "Uses tsfresh MinimalFCParameters on each window before sklearn classification."
    INPUT_REQUIREMENTS: str = "Input must be shaped as (B, C, L) or (B, L, C)."
    ARCHITECTURE_COMPONENTS: ArchitectureSpec = ArchitectureSpec(
        feature_engineering=True,
        classical_ml=True,
    )

    def __init__(
        self,
        *,
        input_channels: int,
        window_length: int,
        num_classes: int,
        feature_parameters: dict[str, Any] | None = None,
        feature_jobs: int = 0,
        disable_progressbar: bool = True,
        **estimator_kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.window_length = window_length
        self.num_classes = num_classes
        self.feature_parameters = feature_parameters or MinimalFCParameters()
        self.feature_jobs = feature_jobs
        self.disable_progressbar = disable_progressbar
        self.estimator_kwargs = estimator_kwargs
        self.estimator = self._build_estimator(**estimator_kwargs)
        self.feature_columns_: list[str] | None = None
        self.architecture = self.get_architecture_components()
        self.register_buffer("_device_ref", torch.empty(0), persistent=False)

    @abstractmethod
    def _build_estimator(self, **estimator_kwargs: Any) -> ClassifierMixin:
        raise NotImplementedError

    def get_name(self) -> str:
        name = getattr(self, "NAME", None)
        if isinstance(name, str) and name.strip():
            return name
        return self.__class__.__name__

    def get_display_name(self) -> str:
        display_name = getattr(self, "display_name", None)
        if isinstance(display_name, str) and display_name.strip():
            return display_name
        return self.get_name()

    def get_color(self) -> str:
        color = getattr(self, "color", None)
        if isinstance(color, str) and color.strip():
            return color
        return "#1f6aa5"

    def get_architecture(self) -> str:
        arch = getattr(self, "ARCHITECTURE", None)
        if isinstance(arch, str) and arch.strip():
            return arch
        return self.get_name()

    def get_architecture_components(self) -> ArchitectureSpec:
        components = getattr(self, "ARCHITECTURE_COMPONENTS", ArchitectureSpec())
        if isinstance(components, ArchitectureSpec):
            return components
        raise TypeError(
            f"{self.__class__.__name__}.ARCHITECTURE_COMPONENTS must be an ArchitectureSpec instance."
        )

    def get_input_type(self) -> str:
        return self.INPUT_TYPE

    def get_source(self) -> str:
        return self.SOURCE

    def get_notes(self) -> str:
        return self.NOTES

    def get_input_requirements(self) -> str:
        return self.INPUT_REQUIREMENTS

    def get_trainable_param_count(self) -> int:
        return 0

    def to_input_shape(self, x: ArrayLike) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            array = x.detach().cpu().numpy()
        else:
            array = np.asarray(x)

        if array.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, got {array.ndim}D input with shape {tuple(array.shape)}")
        if array.shape[1] == self.input_channels:
            array = array.transpose(0, 2, 1)
        elif array.shape[2] != self.input_channels:
            raise ValueError(
                f"3D input must match (B, {self.input_channels}, L) or "
                f"(B, L, {self.input_channels}); got {tuple(array.shape)}"
            )

        return np.asarray(array, dtype=np.float64)

    def _to_long_dataframe(self, x: np.ndarray) -> pd.DataFrame:
        batch_size, seq_len, num_sensors = x.shape
        sample_ids = np.repeat(np.arange(batch_size), seq_len * num_sensors)
        times = np.tile(np.repeat(np.arange(seq_len), num_sensors), batch_size)
        sensor_names = np.tile(np.array([f"sensor_{idx}" for idx in range(num_sensors)]), batch_size * seq_len)

        return pd.DataFrame(
            {
                "id": sample_ids,
                "time": times,
                "kind": sensor_names,
                "value": x.reshape(-1),
            }
        )

    def _extract_feature_frame(self, x: ArrayLike) -> pd.DataFrame:
        x_arr = self.to_input_shape(x)
        long_df = self._to_long_dataframe(x_arr)
        logger.info(f"[{self.get_name()}] Extracting features from {x_arr.shape[0]} samples")
        feature_frame = extract_features(
            long_df,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=self.feature_parameters,
            impute_function=impute,
            disable_progressbar=self.disable_progressbar,
            show_warnings=False,
            n_jobs=self.feature_jobs,
        )
        return feature_frame.sort_index()

    def _normalize_labels(self, y: ArrayLike) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            labels = y.detach().cpu().numpy()
        else:
            labels = np.asarray(y)

        if labels.ndim == 2:
            return labels.argmax(axis=1)
        if labels.ndim != 1:
            raise ValueError(f"Expected labels with shape (B,) or one-hot shape (B, K); got {tuple(labels.shape)}")
        return labels.astype(int)

    def _align_features(self, feature_frame: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit or self.feature_columns_ is None:
            self.feature_columns_ = list(feature_frame.columns)
            return feature_frame
        return feature_frame.reindex(columns=self.feature_columns_, fill_value=0.0)

    def fit(self, x: ArrayLike, y: ArrayLike) -> "ClassicalHARModel":
        logger.info(f"[{self.get_name()}] sklearn version: {sklearn_version}")
        labels = self._normalize_labels(y)
        t0 = time.perf_counter()
        features = self._align_features(self._extract_feature_frame(x), fit=True)
        logger.info(f"[{self.get_name()}] Feature extraction took {time.perf_counter() - t0:.2f}s")

        t0 = time.perf_counter()
        self.estimator.fit(features, labels)
        logger.info(f"[{self.get_name()}] Training took {time.perf_counter() - t0:.2f}s")
        return self

    def forward(self, x: ArrayLike) -> torch.Tensor:
        output_device = x.device if isinstance(x, torch.Tensor) else self._device_ref.device
        scores = self.decision_scores(x)
        return torch.as_tensor(scores, dtype=torch.float32, device=output_device)

    def predict(self, x: ArrayLike) -> np.ndarray:
        features = self._align_features(self._extract_feature_frame(x), fit=False)
        return np.asarray(self.estimator.predict(features))

    def predict_proba(self, x: ArrayLike) -> np.ndarray:
        self._ensure_fitted()
        features = self._align_features(self._extract_feature_frame(x), fit=False)
        predict_proba = getattr(self.estimator, "predict_proba")
        return self._align_estimator_columns(
            np.asarray(predict_proba(features), dtype=np.float64),
            fill_value=0.0,
        )

    def decision_scores(self, x: ArrayLike) -> np.ndarray:
        self._ensure_fitted()
        features = self._align_features(self._extract_feature_frame(x), fit=False)

        predict_proba = getattr(self.estimator, "predict_proba", None)
        if predict_proba is not None:
            try:
                probabilities = np.asarray(predict_proba(features), dtype=np.float64)
            except AttributeError:
                probabilities = None
            if probabilities is not None:
                return self._align_estimator_columns(
                    np.log(np.clip(probabilities, 1e-12, 1.0)),
                    fill_value=-1e9,
                )

        decision_function = getattr(self.estimator, "decision_function", None)
        if decision_function is not None:
            scores = np.asarray(decision_function(features), dtype=np.float64)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            if scores.ndim == 2 and scores.shape[1] == len(self._estimator_classes()):
                return self._align_estimator_columns(scores, fill_value=-1e9)

        predictions = np.asarray(self.estimator.predict(features), dtype=int)
        scores = np.full((predictions.shape[0], self.num_classes), -1e9, dtype=np.float64)
        valid = (predictions >= 0) & (predictions < self.num_classes)
        scores[np.arange(predictions.shape[0])[valid], predictions[valid]] = 0.0
        return scores

    def score(self, x: ArrayLike, y: ArrayLike) -> float:
        predictions = self.predict(x)
        labels = self._normalize_labels(y)
        return float(accuracy_score(labels, predictions))

    def _ensure_fitted(self) -> None:
        if not hasattr(self.estimator, "classes_"):
            raise RuntimeError(f"{self.get_name()} must be fit before inference.")

    def _estimator_classes(self) -> np.ndarray:
        self._ensure_fitted()
        return np.asarray(getattr(self.estimator, "classes_"), dtype=int)

    def _align_estimator_columns(self, values: np.ndarray, *, fill_value: float) -> np.ndarray:
        classes = self._estimator_classes()
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        aligned = np.full((values.shape[0], self.num_classes), fill_value, dtype=np.float64)
        for source_index, class_id in enumerate(classes):
            if 0 <= class_id < self.num_classes and source_index < values.shape[1]:
                aligned[:, class_id] = values[:, source_index]
        return aligned
