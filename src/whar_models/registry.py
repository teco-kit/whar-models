from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class WHARModelID(Enum):
    """Identifiers for all built-in model implementations."""

    AROMA_JOINT_MODEL = "aroma_joint_model"
    ATTEND_DISCRIMINATE = "attend_discriminate"
    ATTENSENSE = "attensense"
    CNN_HAR = "cnn_har"
    DANA = "dana"
    DEEPCONV_LSTM = "deepconv_lstm"
    DEEPCONV_LSTM_ATTENTION = "deepconv_lstm_attention"
    DEEPCONV_LSTM_ISWC = "deepconv_lstm_iswc"
    DEEPSENSE = "deepsense"
    DYNAMIC_WHAR = "dynamic_whar"
    GLOBAL_FUSION = "global_fusion"
    IF_CONV_TRANSFORMER = "if_conv_transformer"
    KNN = "knn"
    LSTMS_ENSEMBLE = "lstms_ensemble"
    MLP_HAR = "mlp_har"
    MLP_MIXER = "mlp_mixer"
    RANDOM_FOREST = "random_forest"
    SA_HAR = "sa_har"
    SVM = "svm"
    TINIERHAR = "tinierhar"
    TINYHAR = "tinyhar"
    TRIPLE_CROSS_DOMAIN_ATTENTION = "triple_cross_domain_attention"


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


_MODEL_SPECS: dict[WHARModelID, ModelSpec] = {
    WHARModelID.AROMA_JOINT_MODEL: ModelSpec(
        id="aroma_joint_model",
        name="AROMA Joint",
        family="neural",
        framework="torch",
        paper="AROMA: A Deep Multi-Task Learning Based Simple and Complex Human Activity Recognition Method Using Wearable Sensors",
        builder=_packaged_builder("aroma_joint_model", "build_aroma_joint_model"),
    ),
    WHARModelID.ATTEND_DISCRIMINATE: ModelSpec(
        id="attend_discriminate",
        name="Attend+Discriminate",
        family="neural",
        framework="torch",
        paper="Attend and Discriminate: Beyond the State-of-the-Art for Human Activity Recognition Using Wearable Sensors",
        builder=_packaged_builder("attend_discriminate", "build_attend_discriminate"),
    ),
    WHARModelID.ATTENSENSE: ModelSpec(
        id="attensense",
        name="AttenSense",
        family="neural",
        framework="torch",
        paper="AttnSense: Multi-level Attention Mechanism for Multimodal Human Activity Recognition",
        builder=_packaged_builder("attensense", "build_attensense"),
    ),
    WHARModelID.CNN_HAR: ModelSpec(
        id="cnn_har",
        name="CNN-HAR",
        family="neural",
        framework="torch",
        paper="Deep Convolutional Neural Networks on Multichannel Time Series for Human Activity Recognition",
        builder=_packaged_builder("cnn_har", "build_cnn_har"),
    ),
    WHARModelID.DANA: ModelSpec(
        id="dana",
        name="DANA",
        family="neural",
        framework="torch",
        paper="DANA: Dimension-Adaptive Neural Architecture for Multivariate Sensor Data",
        builder=_packaged_builder("dana", "build_dana"),
    ),
    WHARModelID.DEEPCONV_LSTM: ModelSpec(
        id="deepconv_lstm",
        name="DeepConvLSTM",
        family="neural",
        framework="torch",
        paper="Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition",
        builder=_packaged_builder("deepconv_lstm", "build_deepconv_lstm"),
    ),
    WHARModelID.DEEPCONV_LSTM_ATTENTION: ModelSpec(
        id="deepconv_lstm_attention",
        name="DeepConvLSTM-Attn",
        family="neural",
        framework="torch",
        paper="On Attention Models for Human Activity Recognition",
        builder=_packaged_builder("deepconv_lstm_attention", "build_deepconv_lstm_attention"),
    ),
    WHARModelID.DEEPCONV_LSTM_ISWC: ModelSpec(
        id="deepconv_lstm_iswc",
        name="DeepConvShallowLSTM",
        family="neural",
        framework="torch",
        paper="Improving Deep Learning for HAR with Shallow LSTMs",
        builder=_packaged_builder("deepconv_lstm_iswc", "build_deepconv_lstm_iswc"),
    ),
    WHARModelID.DEEPSENSE: ModelSpec(
        id="deepsense",
        name="DeepSense",
        family="neural",
        framework="torch",
        paper="DeepSense: A Unified Deep Learning Framework for Time-Series Mobile Sensing Data Processing",
        builder=_packaged_builder("deepsense", "build_deepsense"),
    ),
    WHARModelID.DYNAMIC_WHAR: ModelSpec(
        id="dynamic_whar",
        name="DynamicWHAR",
        family="neural",
        framework="torch",
        paper="Towards a Dynamic Inter-Sensor Correlations Learning Framework for Multi-Sensor-Based Wearable Human Activity Recognition",
        builder=_packaged_builder("dynamic_whar", "build_dynamic_whar"),
    ),
    WHARModelID.GLOBAL_FUSION: ModelSpec(
        id="global_fusion",
        name="GlobalFusion",
        family="neural",
        framework="torch",
        paper="GlobalFusion: A Global Attentional Deep Learning Framework for Multisensor Information Fusion",
        builder=_packaged_builder("global_fusion", "build_global_fusion"),
    ),
    WHARModelID.IF_CONV_TRANSFORMER: ModelSpec(
        id="if_conv_transformer",
        name="IF-ConvTransformer",
        family="neural",
        framework="torch",
        paper="IF-ConvTransformer: A Framework for Human Activity Recognition Using IMU Fusion and ConvTransformer",
        builder=_packaged_builder("if_conv_transformer", "build_if_conv_transformer"),
    ),
    WHARModelID.LSTMS_ENSEMBLE: ModelSpec(
        id="lstms_ensemble",
        name="Guan-LSTM",
        family="neural",
        framework="torch",
        paper="Ensembles of Deep LSTM Learners for Activity Recognition Using Wearables",
        builder=_packaged_builder("lstms_ensemble", "build_lstms_ensemble"),
    ),
    WHARModelID.KNN: ModelSpec(
        id="knn",
        name="k-NN",
        family="classical_ml",
        framework="sklearn",
        paper="Scikit-learn: Machine Learning in Python",
        builder=_packaged_builder("knn", "build_knn"),
    ),
    WHARModelID.MLP_HAR: ModelSpec(
        id="mlp_har",
        name="MLP-HAR",
        family="neural",
        framework="torch",
        paper="MLP-HAR: Boosting Performance and Efficiency of HAR Models on Edge Devices with Purely Fully Connected Layers",
        builder=_packaged_builder("mlp_har", "build_mlp_har"),
    ),
    WHARModelID.MLP_MIXER: ModelSpec(
        id="mlp_mixer",
        name="MLP-Mixer",
        family="neural",
        framework="torch",
        paper="MLPs Are All You Need for Human Activity Recognition",
        builder=_packaged_builder("mlp_mixer", "build_mlp_mixer"),
    ),
    WHARModelID.RANDOM_FOREST: ModelSpec(
        id="random_forest",
        name="Random Forest",
        family="classical_ml",
        framework="sklearn",
        paper="Scikit-learn: Machine Learning in Python",
        builder=_packaged_builder("random_forest", "build_random_forest"),
    ),
    WHARModelID.SA_HAR: ModelSpec(
        id="sa_har",
        name="SA-HAR",
        family="neural",
        framework="torch",
        paper="Human Activity Recognition from Wearable Sensor Data Using Self-Attention",
        builder=_packaged_builder("sa_har", "build_sa_har"),
    ),
    WHARModelID.SVM: ModelSpec(
        id="svm",
        name="SVM",
        family="classical_ml",
        framework="sklearn",
        paper="Scikit-learn: Machine Learning in Python",
        builder=_packaged_builder("svm", "build_svm"),
    ),
    WHARModelID.TINIERHAR: ModelSpec(
        id="tinierhar",
        name="TinierHAR",
        family="neural",
        framework="torch",
        paper="TinierHAR: Towards Ultra-Lightweight Deep Learning Models for Efficient Human Activity Recognition on Edge Devices",
        builder=_packaged_builder("tinierhar", "build_tinierhar"),
    ),
    WHARModelID.TINYHAR: ModelSpec(
        id="tinyhar",
        name="TinyHAR",
        family="compact_neural",
        framework="torch",
        paper="TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition",
        builder=_build_tinyhar,
    ),
    WHARModelID.TRIPLE_CROSS_DOMAIN_ATTENTION: ModelSpec(
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


def _require_model_id(model_id: WHARModelID) -> WHARModelID:
    if not isinstance(model_id, WHARModelID):
        raise TypeError("model_id must be a WHARModelID, for example WHARModelID.TINYHAR")
    return model_id


def list_models() -> list[ModelSpec]:
    return sorted(_MODEL_SPECS.values(), key=lambda spec: spec.id)


def get_model_spec(model_id: WHARModelID) -> ModelSpec:
    return _MODEL_SPECS[_require_model_id(model_id)]


def build_model(
    model_id: WHARModelID,
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
