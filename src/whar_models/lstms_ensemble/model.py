from __future__ import annotations

import torch
from torch import nn

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


class SingleModel(nn.Module):
    def __init__(self, n_channels, n_hidden=256, n_layers=2, n_classes=18, drop_prob=0.5):
        super(SingleModel, self).__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.drop_prob = drop_prob
        self.n_channels = n_channels

        self.lstm = nn.LSTM(n_channels, n_hidden, n_layers, dropout=self.drop_prob)
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden, batch_size):
        x = x.permute(1, 0, 2)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = x.contiguous().view(-1, self.n_hidden)
        out = self.fc(x)

        return out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state."""
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
        )
        return hidden


def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)


class LSTMsEnsemble_Wrapper(ModelWrapper):
    NAME = "LSTMsEnsemble"
    display_name = "Guan-LSTM"
    color = "#c1121f"
    ARCHITECTURE = "2x LSTM + dropout + linear"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(dense=True, recurrent=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/dspanah/Sensor-Based-Human-Activity-Recognition-LSTMsEnsemble-Pytorch"
    NOTES = (
        "Ports the notebook SingleModel exactly. The original repo's ensemble is a training-time "
        "average over top saved checkpoints of this same model, not a separate network. The wrapper "
        "adapts sequence output to benchmark classification by using the last timestep logits."
    )

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        ts_len: int = 128,
        n_hidden: int = 256,
        n_layers: int = 2,
        drop_prob: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(num_sensors=num_sensors, num_classes=num_classes)
        _ = ts_len, kwargs
        self.model = SingleModel(
            n_channels=num_sensors,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_classes=num_classes,
            drop_prob=drop_prob,
        )
        self.model.apply(init_weights)

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] == self.num_sensors:
                return x
            raise ValueError(f"3D input must match (B, L, {self.num_sensors}); got {tuple(x.shape)}")
        raise ValueError(f"Expected 3D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        batch_size = x.shape[0]
        hidden = self.model.init_hidden(batch_size)
        hidden = tuple(each.to(device=x.device, dtype=x.dtype) for each in hidden)
        out, _ = self.model(x, hidden, batch_size)
        out = out.view(x.shape[1], batch_size, self.num_classes)
        return out[-1]


from whar_models._shared.adapter import ChannelFirstAdapter


def build_lstms_ensemble(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        LSTMsEnsemble_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
