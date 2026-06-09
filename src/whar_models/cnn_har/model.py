from __future__ import annotations

import torch
import torch.nn as nn

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


class CNNHARModel(nn.Module):
    """
    PyTorch port of the CNNHAR (MatConvNet) architecture from:
    https://github.com/jianboyang/CNNHAR (fcnn.m)
    """

    def __init__(self, num_sensors: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(5, 1), stride=(1, 1), padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(2, 1), padding=0)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=1e-4 / 5.0, beta=0.75, k=1.0)

        self.conv2 = nn.Conv2d(50, 40, kernel_size=(5, 1), stride=(1, 1), padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(2, 1), padding=0)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=1e-4 / 5.0, beta=0.75, k=1.0)

        self.conv3 = nn.Conv2d(40, 20, kernel_size=(3, 1), stride=(1, 1), padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.norm3 = nn.LocalResponseNorm(size=5, alpha=1e-4 / 5.0, beta=0.75, k=1.0)

        # Fuse across all sensor channels (width dimension).
        self.conv4 = nn.Conv2d(20, 400, kernel_size=(1, num_sensors), stride=(1, 1), padding=0)
        self.relu4 = nn.ReLU(inplace=True)
        self.norm4 = nn.LocalResponseNorm(size=5, alpha=1e-4 / 5.0, beta=0.75, k=1.0)

        # Reference model expects final spatial size 1x1 for classification.
        # Keep compatibility with variable window lengths in this benchmark.
        self.temporal_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Conv2d(400, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.norm4(x)

        x = self.temporal_pool(x)
        x = self.classifier(x)
        return x.squeeze(-1).squeeze(-1)


class CNNHAR(ModelWrapper):
    NAME = "CNNHAR"
    display_name = "CNN-HAR"
    color = "#b35806"
    ARCHITECTURE = "Conv2D + max-pool + LRN + sensor-fusion conv"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(cnn=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/jianboyang/CNNHAR (fcnn.m)"
    NOTES = (
        "PyTorch port of reference fixed-window CNNHAR. "
        "This implementation uses adaptive temporal pooling, so it supports variable ts_len; "
        "a strict fixed-window variant would remove adaptive pooling and accept only one ts_len."
    )

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        ts_len: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(num_sensors=num_sensors, num_classes=num_classes)
        _ = ts_len, kwargs
        self.model = CNNHARModel(num_sensors=num_sensors, num_classes=num_classes)

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] == self.num_sensors:
                x = x.unsqueeze(1)
            else:
                raise ValueError(f"3D input must match (B, L, {self.num_sensors}); got {tuple(x.shape)}")
        else:
            raise ValueError(f"Expected 3D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        return self.model(x)


from whar_models._shared.adapter import ChannelFirstAdapter


def build_cnn_har(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        CNNHAR,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
