from pathlib import Path

import torch

from whar_models import build_model, list_models


def test_tinyhar_builds_and_runs() -> None:
    model = build_model(
        "tinyhar",
        input_channels=6,
        window_length=128,
        num_classes=5,
    )
    y = model(torch.randn(2, 6, 128))

    assert y.shape == (2, 5)


def test_registry_lists_tinyhar() -> None:
    assert "tinyhar" in [spec.id for spec in list_models()]


def test_each_registered_model_has_a_package_folder() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "whar_models"

    for spec in list_models():
        assert (package_root / spec.id / "__init__.py").is_file()
        assert (package_root / spec.id / "model.py").is_file()


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
            model_id,
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

    knn = build_model("knn", input_channels=6, window_length=128, num_classes=5)
    assert knn.estimator.get_params()["n_neighbors"] == 5
    assert knn.estimator.get_params()["weights"] == "distance"

    random_forest = build_model("random_forest", input_channels=6, window_length=128, num_classes=5)
    random_forest_params = random_forest.estimator.get_params()
    assert random_forest_params["n_estimators"] == 100
    assert random_forest_params["random_state"] == 42
    assert random_forest_params["n_jobs"] == -1
    assert random_forest_params["max_depth"] == 8
    assert random_forest_params["min_samples_leaf"] == 10

    svm = build_model("svm", input_channels=6, window_length=128, num_classes=5)
    svm_params = svm.estimator.get_params()
    assert svm_params["kernel"] == "rbf"
    assert svm_params["C"] == 1.0
    assert svm_params["gamma"] == "scale"
    assert svm_params["probability"] is False
