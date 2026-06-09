from __future__ import annotations

import torch

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper
from whar_models.tinierhar.tinierhar_model import DEFAULT_CONFIG, TinierHARModel


class TinierHAR_Wrapper(ModelWrapper):
    NAME = "TinierHAR"
    display_name = "TinierHAR"
    color = "#f9c74f"
    ARCHITECTURE = "Depthwise-separable Conv2D blocks + BiGRU + attention"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(attention=True, cnn=True, dense=True, recurrent=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/zhaxidele/TinierHAR"
    NOTES = "TinierHAR implementation with depthwise-separable convolutions, BiGRU layers, and attention."

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        ts_len: int = 128,
        config: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(num_sensors=num_sensors, num_classes=num_classes)
        model_config = dict(DEFAULT_CONFIG)
        if config is not None:
            model_config.update(config)
        model_config.update(kwargs)

        self.model = TinierHARModel(
            input_shape=(1, 1, ts_len, num_sensors),
            nb_classes=num_classes,
            filter_scaling_factor=model_config["filter_scaling_factor"],
            config=model_config,
        )

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] == self.num_sensors:
                return x.unsqueeze(1)
            if x.shape[1] == self.num_sensors:
                return x.transpose(1, 2).unsqueeze(1)
            raise ValueError(
                f"3D input must match (B, L, {self.num_sensors}) or (B, {self.num_sensors}, L); got {tuple(x.shape)}"
            )
        if x.ndim == 4:
            if x.shape[1] != 1 or x.shape[3] != self.num_sensors:
                raise ValueError(f"4D input must match (B, 1, L, {self.num_sensors}); got {tuple(x.shape)}")
            return x
        raise ValueError(f"Expected 3D or 4D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        return self.model(x)


from whar_models._shared.adapter import ChannelFirstAdapter


def build_tinierhar(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        TinierHAR_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
