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
from whar_models import WHARModelID, list_models

for spec in list_models():
    print(spec.id, spec.name, spec.family, spec.framework)

print(WHARModelID.DEEPCONV_LSTM.value)
```

### PyTorch Model Training and Inference

```python
import torch

from whar_models import WHARModelID, build_model

model = build_model(
    WHARModelID.DEEPCONV_LSTM,
    input_channels=6,
    window_length=128,
    num_classes=5,
)

x_train = torch.randn(32, 6, 128)
y_train = torch.randint(0, 5, size=(32,))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for _ in range(10):
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x_train), y_train)
    loss.backward()
    optimizer.step()

model.eval()
x_test = torch.randn(8, 6, 128)
with torch.no_grad():
    logits = model(x_test)
predictions = logits.argmax(dim=1)
```

### Classical Model Inference After Fitting

```python
import torch

from whar_models import WHARModelID, build_model

model = build_model(
    WHARModelID.RANDOM_FOREST,
    input_channels=6,
    window_length=128,
    num_classes=5,
)

x_train = torch.randn(32, 6, 128)
y_train = torch.randint(0, 5, size=(32,))
model.fit(x_train, y_train)

x_test = torch.randn(8, 6, 128)
scores = model(x_test)
predictions = scores.argmax(dim=1)
```

Classical models accept `torch.Tensor` and `numpy.ndarray` inputs with the same
`(batch, channels, timesteps)` shape as neural models. Tensor inputs are
converted internally for feature extraction and scikit-learn inference. Their
`fit(...)` method fits the underlying estimator; it is not a benchmark training
loop and does not support gradient-based training.

# Supported Models

### Neural Models

| Supported | Name | Model | Paper | Framework |
| --- | --- | --- | --- | --- |
| ✅ | `WHARModelID.AROMA_JOINT_MODEL` | AROMA Joint | [*AROMA: A Deep Multi-Task Learning Based Simple and Complex Human Activity Recognition Method Using Wearable Sensors*](https://dl.acm.org/doi/pdf/10.1145/3214277) | PyTorch |
| ✅ | `WHARModelID.ATTEND_DISCRIMINATE` | Attend+Discriminate | [*Attend and Discriminate: Beyond the State-of-the-Art for Human Activity Recognition Using Wearable Sensors*](https://arxiv.org/pdf/2007.07172) | PyTorch |
| ✅ | `WHARModelID.ATTENSENSE` | AttenSense | [*AttnSense: Multi-level Attention Mechanism For Multimodal Human Activity Recognition*](https://www.ijcai.org/proceedings/2019/0431.pdf) | PyTorch |
| ✅ | `WHARModelID.CNN_HAR` | CNN-HAR | [*Deep Convolutional Neural Networks on Multichannel Time Series for Human Activity Recognition*](https://www.ijcai.org/Proceedings/15/Papers/561.pdf) | PyTorch |
| ✅ | `WHARModelID.DANA` | DANA | [*DANA: Dimension-Adaptive Neural Architecture for Multivariate Sensor Data*](https://arxiv.org/pdf/2008.02397) | PyTorch |
| ✅ | `WHARModelID.DEEPCONV_LSTM` | DeepConvLSTM | [*Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*](https://www.mdpi.com/1424-8220/16/1/115/pdf) | PyTorch |
| ✅ | `WHARModelID.DEEPCONV_LSTM_ATTENTION` | DeepConvLSTM-Attn | [*On Attention Models for Human Activity Recognition*](https://arxiv.org/pdf/1805.07648) | PyTorch |
| ✅ | `WHARModelID.DEEPCONV_LSTM_ISWC` | DeepConvShallowLSTM | [*Improving Deep Learning for HAR with Shallow LSTMs*](https://dl.acm.org/doi/pdf/10.1145/3460421.3480419) | PyTorch |
| ✅ | `WHARModelID.DEEPSENSE` | DeepSense | [*DeepSense: A Unified Deep Learning Framework for Time-Series Mobile Sensing Data Processing*](https://www.shuochao.net/publication/yao-2017-deepsense/yao-2017-deepsense.pdf) | PyTorch |
| ✅ | `WHARModelID.DYNAMIC_WHAR` | DynamicWHAR | [*Towards a Dynamic Inter-Sensor Correlations Learning Framework for Multi-Sensor-Based Wearable Human Activity Recognition*](https://dl.acm.org/doi/pdf/10.1145/3550331) | PyTorch |
| ✅ | `WHARModelID.GLOBAL_FUSION` | GlobalFusion | [*GlobalFusion: A Global Attentional Deep Learning Framework for Multisensor Information Fusion*](https://dl.acm.org/doi/pdf/10.1145/3380999) | PyTorch |
| ✅ | `WHARModelID.IF_CONV_TRANSFORMER` | IF-ConvTransformer | [*IF-ConvTransformer: A Framework for Human Activity Recognition Using IMU Fusion and ConvTransformer*](https://dl.acm.org/doi/pdf/10.1145/3534584) | PyTorch |
| ✅ | `WHARModelID.LSTMS_ENSEMBLE` | Guan-LSTM | [*Ensembles of Deep LSTM Learners for Activity Recognition Using Wearables*](https://arxiv.org/pdf/1703.09370) | PyTorch |
| ✅ | `WHARModelID.MLP_HAR` | MLP-HAR | [*MLP-HAR: Boosting Performance and Efficiency of HAR Models on Edge Devices with Purely Fully Connected Layers*](https://dl.acm.org/doi/pdf/10.1145/3675095.3676624) | PyTorch |
| ✅ | `WHARModelID.MLP_MIXER` | MLP-Mixer | [*MLPs Are All You Need for Human Activity Recognition*](https://www.mdpi.com/2076-3417/13/20/11154/pdf) | PyTorch |
| ✅ | `WHARModelID.SA_HAR` | SA-HAR | [*Human Activity Recognition from Wearable Sensor Data Using Self-Attention*](https://arxiv.org/pdf/2003.09018) | PyTorch |
| ✅ | `WHARModelID.TINIERHAR` | TinierHAR | [*TinierHAR: Towards Ultra-Lightweight Deep Learning Models for Efficient Human Activity Recognition on Edge Devices*](https://arxiv.org/pdf/2507.07949) | PyTorch |
| ✅ | `WHARModelID.TINYHAR` | TinyHAR | [*TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition*](https://dl.acm.org/doi/pdf/10.1145/3544794.3558467) | PyTorch |
| ✅ | `WHARModelID.TRIPLE_CROSS_DOMAIN_ATTENTION` | Triple-Cross-Attn | [*Triple Cross-Domain Attention on Human Activity Recognition Using Wearable Sensors*](https://yinntag.github.io/publications/P3.pdf) | PyTorch |

### Classical Models

| Supported | Name | Model | Paper | Framework | Default Configuration |
| --- | --- | --- | --- | --- | --- |
| ✅ | `WHARModelID.KNN` | k-NN | [*Scikit-learn: Machine Learning in Python*](https://jmlr.csail.mit.edu/papers/volume12/pedregosa11a/pedregosa11a.pdf) | scikit-learn + tsfresh | `n_neighbors=5`, `weights="distance"` |
| ✅ | `WHARModelID.RANDOM_FOREST` | Random Forest | [*Scikit-learn: Machine Learning in Python*](https://jmlr.csail.mit.edu/papers/volume12/pedregosa11a/pedregosa11a.pdf) | scikit-learn + tsfresh | `n_estimators=100`, `random_state=42`, `n_jobs=-1`, `max_depth=8`, `min_samples_leaf=10` |
| ✅ | `WHARModelID.SVM` | SVM | [*Scikit-learn: Machine Learning in Python*](https://jmlr.csail.mit.edu/papers/volume12/pedregosa11a/pedregosa11a.pdf) | scikit-learn + tsfresh | `kernel="rbf"`, `C=1.0`, `gamma="scale"`, `probability=False` |

# Adding a Model

1. Add an implementation under `src/whar_models/<model_id>/`.
2. Register it in `src/whar_models/registry.py`.
3. Add a focused test that builds the model for a generic WHAR input.

Models should accept variable channel counts, sequence lengths, and class counts. Dataset-specific assumptions belong in the dataset package, not here.
