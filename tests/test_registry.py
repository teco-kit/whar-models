from pathlib import Path

import pytest
import torch

from whar_models import WHARModelID, build_model, list_models


def test_tinyhar_builds_and_runs() -> None:
    model = build_model(
        WHARModelID.TINYHAR,
        input_channels=6,
        window_length=128,
        num_classes=5,
    )
    y = model(torch.randn(2, 6, 128))

    assert y.shape == (2, 5)


def test_registry_lists_tinyhar() -> None:
    assert "tinyhar" in [spec.id for spec in list_models()]


def test_model_enum_and_registry_are_in_sync() -> None:
    assert {model_id.value for model_id in WHARModelID} == {spec.id for spec in list_models()}


def test_each_registered_model_has_a_package_folder() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "whar_models"

    for spec in list_models():
        assert (package_root / spec.id / "__init__.py").is_file()
        assert (package_root / spec.id / "model.py").is_file()


def test_each_registered_model_has_paper_metadata() -> None:
    for spec in list_models():
        assert spec.paper


def test_neural_models_build_and_run() -> None:
    expected_ids = {
        "aroma_joint_model",
        "attend_discriminate",
        "attensense",
        "cnn_har",
        "dana",
        "deepconv_lstm",
        "deepconv_lstm_attention",
        "deepconv_lstm_iswc",
        "deepsense",
        "dynamic_whar",
        "global_fusion",
        "if_conv_transformer",
        "lstms_ensemble",
        "mlp_har",
        "mlp_mixer",
        "sa_har",
        "tinierhar",
        "triple_cross_domain_attention",
    }
    listed_ids = {spec.id for spec in list_models()}

    assert expected_ids <= listed_ids

    for model_id in expected_ids:
        model = build_model(
            WHARModelID(model_id),
            input_channels=6,
            window_length=128,
            num_classes=5,
        )
        model.eval()
        with torch.no_grad():
            y = model(torch.randn(2, 6, 128))

        assert y.shape == (2, 5)


def test_classical_models_use_expected_defaults() -> None:
    expected_ids = {"knn", "random_forest", "svm"}
    listed_ids = {spec.id for spec in list_models()}

    assert expected_ids <= listed_ids

    knn = build_model(WHARModelID.KNN, input_channels=6, window_length=128, num_classes=5)
    assert knn.estimator.get_params()["n_neighbors"] == 5
    assert knn.estimator.get_params()["weights"] == "distance"

    random_forest = build_model(WHARModelID.RANDOM_FOREST, input_channels=6, window_length=128, num_classes=5)
    random_forest_params = random_forest.estimator.get_params()
    assert random_forest_params["n_estimators"] == 100
    assert random_forest_params["random_state"] == 42
    assert random_forest_params["n_jobs"] == -1
    assert random_forest_params["max_depth"] == 8
    assert random_forest_params["min_samples_leaf"] == 10

    svm = build_model(WHARModelID.SVM, input_channels=6, window_length=128, num_classes=5)
    svm_params = svm.estimator.get_params()
    assert svm_params["kernel"] == "rbf"
    assert svm_params["C"] == 1.0
    assert svm_params["gamma"] == "scale"
    assert svm_params["probability"] is False


def test_classical_models_are_callable_after_fit() -> None:
    x = torch.randn(6, 2, 16)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    models = [
        build_model(WHARModelID.KNN, input_channels=2, window_length=16, num_classes=3, n_neighbors=1),
        build_model(
            WHARModelID.RANDOM_FOREST,
            input_channels=2,
            window_length=16,
            num_classes=3,
            n_estimators=4,
            min_samples_leaf=1,
        ),
        build_model(WHARModelID.SVM, input_channels=2, window_length=16, num_classes=3),
    ]

    for model in models:
        assert isinstance(model, torch.nn.Module)
        model.fit(x, y)
        model.eval()

        scores = model(x[:3])

        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (3, 3)
        assert scores.device == x.device
        assert scores.requires_grad is False
        assert model.predict(x[:3]).shape == (3,)


def test_classical_model_call_requires_fit() -> None:
    model = build_model(WHARModelID.RANDOM_FOREST, input_channels=2, window_length=16, num_classes=3)

    with pytest.raises(RuntimeError, match="fit before inference"):
        model(torch.randn(2, 2, 16))


def test_build_model_requires_model_enum() -> None:
    with pytest.raises(TypeError, match="WHARModelID"):
        build_model("tinyhar", input_channels=6, window_length=128, num_classes=5)
