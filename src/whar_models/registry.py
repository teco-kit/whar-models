from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    framework: str
    paper: str | None
    builder: Callable[..., object]


def _build_tinyhar(**kwargs: object) -> object:
    from whar_models.tinyhar import build_tinyhar

    return build_tinyhar(**kwargs)


def _build_packaged_model(package_name: str, builder_name: str, **kwargs: object) -> object:
    from importlib import import_module

    model_module = import_module(f"whar_models.{package_name}")
    builder = getattr(model_module, builder_name)
    return builder(**kwargs)


def _packaged_builder(package_name: str, builder_name: str) -> Callable[..., object]:
    def builder(**kwargs: object) -> object:
        return _build_packaged_model(package_name, builder_name, **kwargs)

    return builder


_MODEL_SPECS: dict[str, ModelSpec] = {
    "aroma_joint_model": ModelSpec(
        id="aroma_joint_model",
        name="AROMA Joint",
        family="neural",
        framework="torch",
        paper=None,
        builder=_packaged_builder("aroma_joint_model", "build_aroma_joint_model"),
    ),
    "attend_discriminate": ModelSpec(
        id="attend_discriminate",
        name="Attend+Discriminate",
        family="neural",
        framework="torch",
        paper="Attend and Discriminate: Beyond the State-of-the-Art for Human Activity Recognition Using Wearable Sensors",
        builder=_packaged_builder("attend_discriminate", "build_attend_discriminate"),
    ),
    "attensense": ModelSpec(
        id="attensense",
        name="AttenSense",
        family="neural",
        framework="torch",
        paper="AttnSense: Multi-level Attention Mechanism for Multimodal Human Activity Recognition",
        builder=_packaged_builder("attensense", "build_attensense"),
    ),
    "cnn_har": ModelSpec(
        id="cnn_har",
        name="CNN-HAR",
        family="neural",
        framework="torch",
        paper=None,
        builder=_packaged_builder("cnn_har", "build_cnn_har"),
    ),
    "dana": ModelSpec(
        id="dana",
        name="DANA",
        family="neural",
        framework="torch",
        paper="DANA: Dimension-Adaptive Neural Architecture for Multivariate Sensor Data",
        builder=_packaged_builder("dana", "build_dana"),
    ),
    "deepconv_lstm": ModelSpec(
        id="deepconv_lstm",
        name="DeepConvLSTM",
        family="neural",
        framework="torch",
        paper="Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition",
        builder=_packaged_builder("deepconv_lstm", "build_deepconv_lstm"),
    ),
    "deepconv_lstm_attention": ModelSpec(
        id="deepconv_lstm_attention",
        name="DeepConvLSTM-Attn",
        family="neural",
        framework="torch",
        paper=None,
        builder=_packaged_builder("deepconv_lstm_attention", "build_deepconv_lstm_attention"),
    ),
    "deepconv_lstm_iswc": ModelSpec(
        id="deepconv_lstm_iswc",
        name="DeepConvShallowLSTM",
        family="neural",
        framework="torch",
        paper=None,
        builder=_packaged_builder("deepconv_lstm_iswc", "build_deepconv_lstm_iswc"),
    ),
    "deepsense": ModelSpec(
        id="deepsense",
        name="DeepSense",
        family="neural",
        framework="torch",
        paper="DeepSense: A Unified Deep Learning Framework for Time-Series Mobile Sensing Data Processing",
        builder=_packaged_builder("deepsense", "build_deepsense"),
    ),
    "dynamic_whar": ModelSpec(
        id="dynamic_whar",
        name="DynamicWHAR",
        family="neural",
        framework="torch",
        paper=None,
        builder=_packaged_builder("dynamic_whar", "build_dynamic_whar"),
    ),
    "global_fusion": ModelSpec(
        id="global_fusion",
        name="GlobalFusion",
        family="neural",
        framework="torch",
        paper=None,
        builder=_packaged_builder("global_fusion", "build_global_fusion"),
    ),
    "if_conv_transformer": ModelSpec(
        id="if_conv_transformer",
        name="IF-ConvTransformer",
        family="neural",
        framework="torch",
        paper="IF-ConvTransformer: A Framework for Human Activity Recognition Using IMU Fusion and ConvTransformer",
        builder=_packaged_builder("if_conv_transformer", "build_if_conv_transformer"),
    ),
    "lstms_ensemble": ModelSpec(
        id="lstms_ensemble",
        name="Guan-LSTM",
        family="neural",
        framework="torch",
        paper=None,
        builder=_packaged_builder("lstms_ensemble", "build_lstms_ensemble"),
    ),
    "knn": ModelSpec(
        id="knn",
        name="k-NN",
        family="classical_ml",
        framework="sklearn",
        paper=None,
        builder=_packaged_builder("knn", "build_knn"),
    ),
    "mlp_har": ModelSpec(
        id="mlp_har",
        name="MLP-HAR",
        family="neural",
        framework="torch",
        paper=None,
        builder=_packaged_builder("mlp_har", "build_mlp_har"),
    ),
    "mlp_mixer": ModelSpec(
        id="mlp_mixer",
        name="MLP-Mixer",
        family="neural",
        framework="torch",
        paper="MLP-Mixer: An all-MLP Architecture for Vision",
        builder=_packaged_builder("mlp_mixer", "build_mlp_mixer"),
    ),
    "random_forest": ModelSpec(
        id="random_forest",
        name="Random Forest",
        family="classical_ml",
        framework="sklearn",
        paper=None,
        builder=_packaged_builder("random_forest", "build_random_forest"),
    ),
    "sa_har": ModelSpec(
        id="sa_har",
        name="SA-HAR",
        family="neural",
        framework="torch",
        paper=None,
        builder=_packaged_builder("sa_har", "build_sa_har"),
    ),
    "svm": ModelSpec(
        id="svm",
        name="SVM",
        family="classical_ml",
        framework="sklearn",
        paper=None,
        builder=_packaged_builder("svm", "build_svm"),
    ),
    "tinierhar": ModelSpec(
        id="tinierhar",
        name="TinierHAR",
        family="neural",
        framework="torch",
        paper="TinierHAR: A Lightweight Deep Learning Model for Human Activity Recognition",
        builder=_packaged_builder("tinierhar", "build_tinierhar"),
    ),
    "tinyhar": ModelSpec(
        id="tinyhar",
        name="TinyHAR",
        family="compact_neural",
        framework="torch",
        paper="TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition",
        builder=_build_tinyhar,
    ),
    "triple_cross_domain_attention": ModelSpec(
        id="triple_cross_domain_attention",
        name="Triple-Cross-Attn",
        family="neural",
        framework="torch",
        paper="Triple Cross-domain Attention for Human Activity Recognition",
        builder=_packaged_builder(
            "triple_cross_domain_attention",
            "build_triple_cross_domain_attention",
        ),
    ),
}


def list_models() -> list[ModelSpec]:
    return sorted(_MODEL_SPECS.values(), key=lambda spec: spec.id)


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return _MODEL_SPECS[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(_MODEL_SPECS))
        raise KeyError(f"Unknown WHAR model '{model_id}'. Known models: {known}") from exc


def build_model(
    model_id: str,
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
) -> object:
    spec = get_model_spec(model_id)
    return spec.builder(
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
