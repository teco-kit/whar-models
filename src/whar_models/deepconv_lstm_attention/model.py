from __future__ import annotations

import torch
import torch.nn as nn

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


DEFAULT_CONFIG = {
    "num_filters": 64,
    "filter_size": 5,
    "filter_stride": 1,
    "num_units_lstm": 128,
    "num_layers_lstm": 2,
    "is_bidirectional": False,
    "dropout": 0.5,
    "attention_dropout": 0.5,
}


class DeepConvLSTMAttentionModel(nn.Module):
    """
    Port of RCNN from:
    https://bitbucket.org/vmurahari3/deepconvlstmattention/src/master/main_script.py
    """

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        num_filters: int = 64,
        filter_size: int = 5,
        filter_stride: int = 1,
        num_units_lstm: int = 128,
        num_layers_lstm: int = 2,
        is_bidirectional: bool = False,
        dropout: float = 0.5,
        attention_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_directions = 2 if is_bidirectional else 1
        self.hidden_size = num_units_lstm
        self.num_layers = num_layers_lstm

        hidden_dim = num_units_lstm * self.num_directions
        lstm_input_size = num_sensors * num_filters

        self.conv2DLayer1 = nn.Conv2d(1, num_filters, (filter_size, 1), stride=(filter_stride, 1))
        self.relu1 = nn.ReLU()
        self.conv2DLayer2 = nn.Conv2d(
            num_filters, num_filters, (filter_size, 1), stride=(filter_stride, 1)
        )
        self.relu2 = nn.ReLU()
        self.conv2DLayer3 = nn.Conv2d(
            num_filters, num_filters, (filter_size, 1), stride=(filter_stride, 1)
        )
        self.relu3 = nn.ReLU()
        self.conv2DLayer4 = nn.Conv2d(
            num_filters, num_filters, (filter_size, 1), stride=(filter_stride, 1)
        )
        self.relu4 = nn.ReLU()

        self.lstm = nn.LSTM(
            lstm_input_size,
            num_units_lstm,
            num_layers_lstm,
            bidirectional=is_bidirectional,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dense_layer = nn.Linear(hidden_dim, num_classes)

        self.attentionLayer1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.attentionLayer2 = nn.Linear(hidden_dim, 1)
        self.softmax_attention = nn.Softmax(dim=0)

    def _init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
        h0 = torch.randn(shape, device=device, dtype=dtype) * 0.08
        c0 = torch.randn(shape, device=device, dtype=dtype) * 0.08
        return h0, c0

    def forward(self, x: torch.Tensor, vis_attention: bool = False):
        convout1 = self.relu1(self.conv2DLayer1(x))
        convout2 = self.relu2(self.conv2DLayer2(convout1))
        convout3 = self.relu3(self.conv2DLayer3(convout2))
        convout4 = self.relu4(self.conv2DLayer4(convout3))

        # Original code shape flow:
        # (B, F, T, C) -> (T, B, F, C) -> (T, B, F*C)
        lstm_input = convout4.permute(2, 0, 1, 3).contiguous()
        lstm_input = lstm_input.view(lstm_input.shape[0], lstm_input.shape[1], -1)
        lstm_input = self.dropout(lstm_input)

        output, _ = self.lstm(
            lstm_input, self._init_hidden(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        )

        # Attention over past timesteps with the last timestep as query-like additive term.
        past_context = output[:-1]
        current = output[-1]
        attention_layer1_output = self.attentionLayer1(past_context) + current
        attention_layer1_output = self.tanh1(attention_layer1_output)
        attention_layer1_output = self.attention_dropout(attention_layer1_output)
        attention_layer2_output = self.attentionLayer2(attention_layer1_output).squeeze(2)
        attn_weights = self.softmax_attention(attention_layer2_output).unsqueeze(2)

        new_context_vector = torch.sum(attn_weights * past_context, 0)
        new_context_vector = new_context_vector + current
        logits = self.dense_layer(new_context_vector)

        if vis_attention:
            return logits, attn_weights
        return logits

    def predict(self, x: torch.Tensor, vis_attention: bool = False):
        if vis_attention:
            logits, attn_weights = self.forward(x, vis_attention=True)
        else:
            logits = self.forward(x, vis_attention=False)

        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs, dim=1)
        if vis_attention:
            return idx, attn_weights
        return idx


class DeepConvLSTMAttention_Wrapper(ModelWrapper):
    NAME = "DeepConvLSTMAttention"
    display_name = "DeepConvLSTM-Attn"
    color = "#fdb863"
    ARCHITECTURE = "4×Conv2D + LSTM + additive temporal attention"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(attention=True, cnn=True, dense=True, recurrent=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://bitbucket.org/vmurahari3/deepconvlstmattention/src/master/main_script.py"
    NOTES = "Keeps original RCNN architecture; adapted for dynamic batch size and benchmark wrapper API."

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        ts_len: int = 128,
        config: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(num_sensors=num_sensors, num_classes=num_classes)
        _ = ts_len
        model_config = dict(DEFAULT_CONFIG)
        if config is not None:
            model_config.update(config)
        model_config.update(kwargs)
        self.model = DeepConvLSTMAttentionModel(
            num_sensors=num_sensors, num_classes=num_classes, **model_config
        )

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


def build_deepconv_lstm_attention(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        DeepConvLSTMAttention_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
