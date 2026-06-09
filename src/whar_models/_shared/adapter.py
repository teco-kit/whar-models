from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    import torch


class ChannelFirstAdapter(nn.Module):
    """Expose benchmark-style wrappers through the WHAR Models input contract."""

    def __init__(
        self,
        model_class: type[nn.Module],
        *,
        input_channels: int,
        window_length: int,
        num_classes: int,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.model = model_class(
            num_sensors=input_channels,
            num_classes=num_classes,
            ts_len=window_length,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("WHAR models expect input shaped (batch, channels, timesteps)")
        return self.model(x.transpose(1, 2))
