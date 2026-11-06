"""Black-box checks that run for every registered model.

Each test is parametrized over all model IDs, so a contributor working on a
single model can run the full check suite for just their model:

    pytest tests/test_single_model.py -k tinierhar

The tests only use the public API (`build_model`, `list_models`) and treat
each model as a black box: they never inspect internal layers, weights, or
training behavior.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from whar_models import ModelSpec, WHARModelID, build_model, get_model_spec, list_models

MODEL_IDS = list(WHARModelID)

BATCH_SIZE = 2
INPUT_CHANNELS = 6
WINDOW_LENGTH = 128
NUM_CLASSES = 5

# Models whose architecture imposes structural constraints beyond the standard
# (batch, channels, timesteps) contract can declare alternate shapes here that
# are known to be valid. All other models are checked against the defaults.
DEFAULT_ALTERNATE_SHAPES = [
    (4, 6, 200),
    (1, 6, 64),
]
ALTERNATE_SHAPES: dict[str, list[tuple[int, int, int]]] = {}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize every test in this module over all model IDs.

    Test IDs are set to the model id string (e.g. ``tinierhar``) so that
    ``pytest -k <model_id>`` selects exactly one model's checks.
    """
    if "model_id" in metafunc.fixturenames:
        metafunc.parametrize("model_id", MODEL_IDS, ids=[m.value for m in MODEL_IDS])


def _spec(model_id: WHARModelID) -> ModelSpec:
    return get_model_spec(model_id)


# ---------------------------------------------------------------------------
# A. Buildability
# ---------------------------------------------------------------------------


def test_build_model(model_id: WHARModelID) -> None:
    """build_model succeeds with standard arguments and returns an nn.Module."""
    model = build_model(
        model_id,
        input_channels=INPUT_CHANNELS,
        window_length=WINDOW_LENGTH,
        num_classes=NUM_CLASSES,
    )
    assert isinstance(model, nn.Module), (
        f"{model_id.value}: build_model must return an nn.Module, got {type(model)!r}"
    )


def test_build_model_is_repeatable(model_id: WHARModelID) -> None:
    """Two builds yield independent instances (no shared module state)."""
    first = build_model(
        model_id,
        input_channels=INPUT_CHANNELS,
        window_length=WINDOW_LENGTH,
        num_classes=NUM_CLASSES,
    )
    second = build_model(
        model_id,
        input_channels=INPUT_CHANNELS,
        window_length=WINDOW_LENGTH,
        num_classes=NUM_CLASSES,
    )
    assert first is not second


# ---------------------------------------------------------------------------
# B. Input/output contract
# ---------------------------------------------------------------------------


def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(x)
    finally:
        model.train(was_training)


def _ready_model(
    model_id: WHARModelID,
    input_channels: int = INPUT_CHANNELS,
    window_length: int = WINDOW_LENGTH,
) -> nn.Module:
    """Build a model ready for inference using only public API calls.

    Classical models must be fitted before they support inference, so they are
    fitted on random data here. This is black-box usage of the documented API.
    """
    spec = _spec(model_id)
    model = build_model(
        model_id,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=NUM_CLASSES,
    )
    if spec.framework == "sklearn":
        model.fit(
            torch.randn(BATCH_SIZE * 4, input_channels, window_length),
            torch.randint(0, NUM_CLASSES, (BATCH_SIZE * 4,)),
        )
    return model


def test_forward_output_shape(model_id: WHARModelID) -> None:
    """(batch, channels, timesteps) input produces (batch, num_classes) output."""
    model = _ready_model(model_id)
    out = _forward(model, torch.randn(BATCH_SIZE, INPUT_CHANNELS, WINDOW_LENGTH))
    assert isinstance(out, torch.Tensor), (
        f"{model_id.value}: forward must return a torch.Tensor, got {type(out)!r}"
    )
    assert tuple(out.shape) == (BATCH_SIZE, NUM_CLASSES), (
        f"{model_id.value}: expected output shape {(BATCH_SIZE, NUM_CLASSES)}, "
        f"got {tuple(out.shape)}"
    )


@pytest.mark.parametrize(
    "batch_size, channels, window",
    DEFAULT_ALTERNATE_SHAPES,
)
def test_forward_other_shapes(
    model_id: WHARModelID, batch_size: int, channels: int, window: int
) -> None:
    """Models must honor the constructor args instead of hardcoding shapes.

    Models with architecture-imposed shape constraints can declare their valid
    alternate shapes in ALTERNATE_SHAPES; those are checked instead of the
    defaults.
    """
    # Models with constraints replace the parametrized shape with their declared
    # shapes; this runs the declared list once per parametrized case, which is
    # acceptable given how few models should ever need an entry here.
    shapes = ALTERNATE_SHAPES.get(model_id.value, [(batch_size, channels, window)])
    for batch_size_i, channels_i, window_i in shapes:
        model = _ready_model(model_id, input_channels=channels_i, window_length=window_i)
        out = _forward(model, torch.randn(batch_size_i, channels_i, window_i))
        assert tuple(out.shape) == (batch_size_i, NUM_CLASSES), (
            f"{model_id.value}: expected output shape {(batch_size_i, NUM_CLASSES)}, "
            f"got {tuple(out.shape)} for input channels={channels_i}, window={window_i}"
        )


def test_forward_rejects_wrong_ndim(model_id: WHARModelID) -> None:
    """Inputs that do not match the (batch, channels, timesteps) contract fail loudly."""
    model = _ready_model(model_id)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        _forward(model, torch.randn(BATCH_SIZE, WINDOW_LENGTH))  # 2D: missing channel dim


def test_eval_mode_deterministic(model_id: WHARModelID) -> None:
    """In eval mode, repeated forward passes on the same input are identical.

    The torch RNG is reset before each pass, so models that legitimately draw
    random numbers during forward (e.g. stochastic hidden-state init) are held
    to the reproducibility contract instead of failing spuriously.
    """
    model = _ready_model(model_id)
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, WINDOW_LENGTH)

    def seeded_forward() -> torch.Tensor:
        torch.manual_seed(0)
        return _forward(model, x)

    first = seeded_forward()
    second = seeded_forward()
    assert torch.equal(first, second), (
        f"{model_id.value}: eval-mode forward passes are not deterministic; "
        "check for dropout/batchnorm left active or stochastic layers"
    )


# ---------------------------------------------------------------------------
# E. Metadata & conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["name", "family", "framework", "paper"],
)
def test_spec_metadata_complete(model_id: WHARModelID, field: str) -> None:
    """Every registered spec carries complete metadata."""
    value: Any = getattr(_spec(model_id), field)
    assert isinstance(value, str) and value.strip(), (
        f"{model_id.value}: registry spec field {field!r} must be a non-empty string"
    )


def test_spec_matches_registry_entry(model_id: WHARModelID) -> None:
    """get_model_spec and list_models agree for this model."""
    spec_ids = [spec.id for spec in list_models()]
    assert model_id.value in spec_ids
    assert _spec(model_id).id == model_id.value


def _wrapped_module(model: nn.Module) -> Any:
    """Return the innermost wrapped module of an adapter chain, if any."""
    current: Any = model
    seen: set[int] = set()
    while not hasattr(current, "NAME") or current.NAME is None:
        if id(current) in seen:
            return current
        seen.add(id(current))
        if hasattr(current, "model") and current.model is not current:
            current = current.model
        else:
            return current
    return current


@pytest.mark.parametrize(
    "attr",
    ["NAME", "ARCHITECTURE", "INPUT_TYPE"],
)
def test_wrapper_metadata_present(model_id: WHARModelID, attr: str) -> None:
    """Built models expose the wrapper metadata used by benchmark tooling."""
    model = _ready_model(model_id)
    wrapped = _wrapped_module(model)
    value = getattr(wrapped, attr, None)
    assert isinstance(value, str) and value.strip(), (
        f"{model_id.value}: built model does not expose non-empty wrapper attribute "
        f"{attr!r} (got {value!r})"
    )
