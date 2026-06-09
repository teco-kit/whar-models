import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.block = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                padding=2,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TinyHAR(nn.Module):
    """Compact TinyHAR-style baseline for multivariate WHAR windows.

    Input shape is `(batch, channels, timesteps)`. The implementation keeps the
    benchmark-facing contract stable while avoiding dataset-specific assumptions.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        window_length: int,
        num_classes: int,
        width: int = 64,
        depth: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        del window_length
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.Sequential(*(ConvBlock(width) for _ in range(depth)))
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TinyHAR expects input shaped (batch, channels, timesteps)")
        x = self.stem(x)
        x = self.encoder(x)
        x = self.temporal_pool(x)
        return self.classifier(x)


def build_tinyhar(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
) -> TinyHAR:
    return TinyHAR(
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
