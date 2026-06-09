from __future__ import annotations

"""
Reference implementation:
https://github.com/drewanye/har-joint-model
"""

import torch
import torch.nn as nn

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


def _init_tf1_style(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _InputBatchNorm(nn.Module):
    """
    Exact port of utils.input_batch_norm from the reference TensorFlow code:
    per-channel normalization using batch+spatial moments and learned affine params.
    """

    def __init__(self, channels: int, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = float(eps)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class _ResidualUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, norm: bool) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.1) if norm else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            bias=True,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.1) if norm else nn.Identity()
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0, bias=True)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

        self.apply(_init_tf1_style)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class _SimpleActivity(nn.Module):
    def __init__(self, *, channels: int, s_num_classes: int, norm: bool) -> None:
        super().__init__()
        self.input_bn = _InputBatchNorm(channels, eps=1e-3)

        self.res1 = _ResidualUnit(channels, 32, norm=norm)
        self.res2 = _ResidualUnit(32, 32, norm=norm)
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)

        self.res3 = _ResidualUnit(32, 64, norm=norm)
        self.res4 = _ResidualUnit(64, 64, norm=norm)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0)

        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.scores = nn.Linear(128, s_num_classes)
        self.relu = nn.ReLU(inplace=True)

        _init_tf1_style(self.fc2)
        _init_tf1_style(self.scores)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_bn(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool0(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.pool1(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        features = self.relu(self.fc2(x))
        logits_s = self.scores(features)
        return features, logits_s


class AROMAJointModel(ModelWrapper):
    NAME = "AROMAJointModel"
    display_name = "AROMA Joint"
    color = "#f6bd60"
    ARCHITECTURE = "Shared residual CNN + 3xLSTM"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(cnn=True, dense=True, recurrent=True, residual=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/drewanye/har-joint-model"
    NOTES = (
        "TensorFlow-to-PyTorch adaptation of the upstream Huynh configuration. "
        "This implementation is not fully faithful to the source model because it was modified to accept "
        "arbitrary benchmark input shapes. The wrapper returns the complex-activity logits. "
        "It preserves the translated per-segment CNN, but the segment MLP now uses "
        "LazyLinear so the per-segment window size can adapt to the first input shape. "
        "If ts_len is divisible by s_win_size, the sequence is split into equal segments. "
        "If ts_len is not divisible by s_win_size, the full sequence is processed as a single segment."
    )
    INPUT_REQUIREMENTS = (
        "Accepts any ts_len. If ts_len is divisible by s_win_size, the sequence is split into equal segments; "
        "otherwise the full sequence is used as a single segment."
    )

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        ts_len: int = 750,
        *,
        c_win_size: int = 15,
        s_win_size: int = 50,
        s_num_classes: int | None = None,
        norm: bool = False,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(num_sensors=num_sensors, num_classes=num_classes)
        _ = kwargs

        self.c_win_size = int(c_win_size)
        self.s_win_size = int(s_win_size)
        self.s_num_classes = int(num_classes if s_num_classes is None else s_num_classes)

        if self.s_win_size < 1:
            raise ValueError(f"s_win_size must be >= 1, got {self.s_win_size}.")

        if ts_len % self.s_win_size == 0:
            inferred_c_win_size = ts_len // self.s_win_size
            if self.c_win_size != inferred_c_win_size:
                self.c_win_size = inferred_c_win_size
        else:
            self.s_win_size = int(ts_len)
            self.c_win_size = 1

        self.simple_activity = _SimpleActivity(
            channels=num_sensors,
            s_num_classes=self.s_num_classes,
            norm=bool(norm),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=int(lstm_hidden_size),
            num_layers=int(lstm_layers),
            batch_first=False,
        )
        self.scores_c = nn.Linear(int(lstm_hidden_size), num_classes)
        _init_tf1_style(self.scores_c)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] == self.num_sensors:
                return x
            if x.shape[1] == self.num_sensors:
                return x.transpose(1, 2)
            raise ValueError(
                f"3D input must match (B, L, {self.num_sensors}) or (B, {self.num_sensors}, L); got {tuple(x.shape)}"
            )
        raise ValueError(f"Expected 3D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        batch_size, ts_len, num_sensors = x.shape
        expected_ts_len = self.c_win_size * self.s_win_size
        if ts_len != expected_ts_len:
            raise ValueError(f"Expected ts_len {expected_ts_len}, got {ts_len}")
        if num_sensors != self.num_sensors:
            raise ValueError(f"Expected {self.num_sensors} sensors, got {num_sensors}")

        x = x.reshape(batch_size, self.c_win_size, self.s_win_size, num_sensors)
        segments = x.reshape(batch_size * self.c_win_size, self.s_win_size, num_sensors)
        segments = segments.transpose(1, 2).unsqueeze(2)

        features, _ = self.simple_activity(segments)
        features = features.reshape(batch_size, self.c_win_size, 128)

        sequence = features.transpose(0, 1).contiguous()
        outputs, _ = self.lstm(sequence)
        return self.scores_c(outputs[-1])


from whar_models._shared.adapter import ChannelFirstAdapter


def build_aroma_joint_model(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        AROMAJointModel,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
