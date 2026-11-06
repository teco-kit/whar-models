from __future__ import annotations

import torch

from whar_models._shared.adapter import ChannelFirstAdapter
from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper
from whar_models.liteway.liteway_full_model import LITEWAY as LITEWAYModel


class _LITEWAYWrapper(ModelWrapper):
    MODEL_CLASS: type
    MODEL_CONFIG: dict[str, int | float]

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        ts_len: int = 128,
        config: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(num_sensors=num_sensors, num_classes=num_classes)
        if ts_len < 4:
            raise ValueError(f"LITEWAY requires at least 4 time steps, got {ts_len}")

        model_config = dict(self.MODEL_CONFIG)
        if config is not None:
            model_config.update(config)
        model_config.update(kwargs)
        self.model = self.MODEL_CLASS(
            input_shape=(1, 1, ts_len, num_sensors),
            nb_classes=num_classes,
            config=model_config,
        )

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] == self.num_sensors:
                return x.unsqueeze(1)
            if x.shape[1] == self.num_sensors:
                return x.transpose(1, 2).unsqueeze(1)
            raise ValueError(
                f"3D input must match (B, L, {self.num_sensors}) or "
                f"(B, {self.num_sensors}, L); got {tuple(x.shape)}"
            )
        if x.ndim == 4:
            if x.shape[1] != 1 or x.shape[3] != self.num_sensors:
                raise ValueError(
                    f"4D input must match (B, 1, L, {self.num_sensors}); got {tuple(x.shape)}"
                )
            if x.shape[2] < 4:
                raise ValueError(f"LITEWAY requires at least 4 time steps, got {x.shape[2]}")
            return x
        raise ValueError(
            f"Expected 3D or 4D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.to_input_shape(x))


class LITEWAY(_LITEWAYWrapper):
    NAME = "LITEWAY"
    display_name = "LITEWAY Full"
    color = "#277da1"
    ARCHITECTURE = "Depthwise-separable CNN + highway temporal convolutions + attention pooling"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(attention=True, cnn=True, dense=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/dominique-nshimyimana/liteway"
    NOTES = "Upstream LITEWAY Full implementation; wrapper adds input shape conversion and class count configuration."
    INPUT_REQUIREMENTS = "Input is (B, L, sensors); L must be at least 4."
    MODEL_CLASS = LITEWAYModel
    MODEL_CONFIG = {"nb_conv_blocks": 4, "nb_filters": 4}


def build_liteway(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        LITEWAY,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
