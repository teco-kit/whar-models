from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class WHARModelBuilder(Protocol):
    def __call__(
        self,
        *,
        input_channels: int,
        window_length: int,
        num_classes: int,
        **kwargs: object,
    ) -> "torch.nn.Module":
        ...
