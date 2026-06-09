# WHAR Models

This library offers reusable model implementations for WHAR (Wearable Human Activity Recognition), including:

- standalone neural architectures for wearable activity recognition
- standalone classical machine-learning baselines
- a unified registry for discovering and constructing models
- consistent model folders under `src/whar_models/<model_id>/`
- a shared input convention for WHAR windows shaped as `(batch, channels, timesteps)`

The library currently includes out-of-the-box support for 22 models. Model-specific defaults are defined in each model package, and overrides can be passed through `build_model`.

# How to Use

### Installation

```
pip install "git+https://github.com/<owner>/whar-models.git"
```

### List Models

```python
from whar_models import list_models

for spec in list_models():
    print(spec.id, spec.name, spec.family, spec.framework)
```

### Example with PyTorch Models

```python
import torch

from whar_models import build_model

model = build_model(
    "deepconv_lstm",
    input_channels=6,
    window_length=128,
    num_classes=5,
)

x = torch.randn(32, 6, 128)
logits = model(x)
```

### Example with Classical Models

```python
import numpy as np

from whar_models import build_model

model = build_model(
    "random_forest",
    input_channels=6,
    window_length=128,
    num_classes=5,
)

x_train = np.random.randn(32, 6, 128)
y_train = np.random.randint(0, 5, size=32)
model.fit(x_train, y_train)

x_test = np.random.randn(8, 6, 128)
predictions = model.predict(x_test)
```

# Supported Models

### Neural Models

| Supported | Model ID | Name | Framework |
| --- | --- | --- | --- |
| Yes | `aroma_joint_model` | AROMA Joint | PyTorch |
| Yes | `attend_discriminate` | Attend+Discriminate | PyTorch |
| Yes | `attensense` | AttenSense | PyTorch |
| Yes | `cnn_har` | CNN-HAR | PyTorch |
| Yes | `dana` | DANA | PyTorch |
| Yes | `deepconv_lstm` | DeepConvLSTM | PyTorch |
| Yes | `deepconv_lstm_attention` | DeepConvLSTM-Attn | PyTorch |
| Yes | `deepconv_lstm_iswc` | DeepConvShallowLSTM | PyTorch |
| Yes | `deepsense` | DeepSense | PyTorch |
| Yes | `dynamic_whar` | DynamicWHAR | PyTorch |
| Yes | `global_fusion` | GlobalFusion | PyTorch |
| Yes | `if_conv_transformer` | IF-ConvTransformer | PyTorch |
| Yes | `lstms_ensemble` | Guan-LSTM | PyTorch |
| Yes | `mlp_har` | MLP-HAR | PyTorch |
| Yes | `mlp_mixer` | MLP-Mixer | PyTorch |
| Yes | `sa_har` | SA-HAR | PyTorch |
| Yes | `tinierhar` | TinierHAR | PyTorch |
| Yes | `tinyhar` | TinyHAR | PyTorch |
| Yes | `triple_cross_domain_attention` | Triple-Cross-Attn | PyTorch |

### Classical Models

| Supported | Model ID | Name | Framework | Default Configuration |
| --- | --- | --- | --- | --- |
| Yes | `knn` | k-NN | scikit-learn + tsfresh | `n_neighbors=5`, `weights="distance"` |
| Yes | `random_forest` | Random Forest | scikit-learn + tsfresh | `n_estimators=100`, `random_state=42`, `n_jobs=-1`, `max_depth=8`, `min_samples_leaf=10` |
| Yes | `svm` | SVM | scikit-learn + tsfresh | `kernel="rbf"`, `C=1.0`, `gamma="scale"`, `probability=False` |

# Adding a Model

1. Add an implementation under `src/whar_models/<model_id>/`.
2. Register it in `src/whar_models/registry.py`.
3. Add a focused test that builds the model for a generic WHAR input.

Models should accept variable channel counts, sequence lengths, and class counts. Dataset-specific assumptions belong in the dataset package, not here.
