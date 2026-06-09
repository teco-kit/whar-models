from __future__ import annotations

import warnings

import torch
from torch import nn

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


warnings.filterwarnings("ignore")


DEFAULT_CONFIG = {
    "no_lstm": False,
    "pooling": False,
    "reduce_layer": False,
    "reduce_layer_output": 8,
    "pool_type": "max",
    "pool_kernel_width": 2,
    "window_size": 128,
    "nb_channels": 6,
    "nb_classes": 5,
    "nb_units_lstm": 128,
    "nb_layers_lstm": 1,
    "nb_conv_blocks": 2,
    "conv_block_type": "normal",
    "nb_filters": 64,
    "filter_width": 11,
    "dilation": 1,
    "batch_norm": False,
    "drop_prob": 0.5,
    "weights_init": "xavier_normal",
    "seed": 1,
}


class ConvBlockFixup(nn.Module):
    def __init__(self, filter_width, input_filters, nb_filters, dilation):
        super().__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(
            self.input_filters,
            self.nb_filters,
            (self.filter_width, 1),
            dilation=(self.dilation, 1),
            bias=False,
            padding="same",
        )
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(
            self.nb_filters,
            self.nb_filters,
            (self.filter_width, 1),
            dilation=(self.dilation, 1),
            bias=False,
            padding="same",
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out += identity
        out = self.relu(out)

        return out


class ConvBlockSkip(nn.Module):
    def __init__(self, window_size, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super().__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(
            self.input_filters,
            self.nb_filters,
            (self.filter_width, 1),
            dilation=(self.dilation, 1),
            padding="same",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            self.nb_filters,
            self.nb_filters,
            (self.filter_width, 1),
            dilation=(self.dilation, 1),
            padding="same",
        )
        self.seq_len = window_size - (filter_width + 1) * 2
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        identity = x
        if self.batch_norm:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.norm1(out)
            out = self.conv2(out)
            out += identity
            out = self.relu(out)
            out = self.norm2(out)
        else:
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            out += identity
            out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, filter_width, input_filters, nb_filters, dilation, batch_norm):
        super().__init__()
        self.filter_width = filter_width
        self.input_filters = input_filters
        self.nb_filters = nb_filters
        self.dilation = dilation
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(
            self.input_filters,
            self.nb_filters,
            (self.filter_width, 1),
            dilation=(self.dilation, 1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            self.nb_filters,
            self.nb_filters,
            (self.filter_width, 1),
            dilation=(self.dilation, 1),
        )
        if self.batch_norm:
            self.norm1 = nn.BatchNorm2d(self.nb_filters)
            self.norm2 = nn.BatchNorm2d(self.nb_filters)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm1(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm2(out)
        return out


class DeepConvLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.no_lstm = config["no_lstm"]
        self.pooling = config["pooling"]
        self.reduce_layer = config["reduce_layer"]
        self.reduce_layer_output = config["reduce_layer_output"]
        self.pool_type = config["pool_type"]
        self.pool_kernel_width = config["pool_kernel_width"]
        self.window_size = config["window_size"]
        self.drop_prob = config["drop_prob"]
        self.nb_channels = config["nb_channels"]
        self.nb_classes = config["nb_classes"]
        self.weights_init = config["weights_init"]
        self.seed = config["seed"]
        self.nb_conv_blocks = config["nb_conv_blocks"]
        self.conv_block_type = config["conv_block_type"]
        self.use_fixup = self.conv_block_type == "fixup"
        self.nb_filters = config["nb_filters"]
        self.filter_width = config["filter_width"]
        self.dilation = config["dilation"]
        self.batch_norm = config["batch_norm"]
        self.nb_units_lstm = config["nb_units_lstm"]
        self.nb_layers_lstm = config["nb_layers_lstm"]

        self.conv_blocks = []
        for i in range(self.nb_conv_blocks):
            if i == 0:
                input_filters = 1
            else:
                input_filters = self.nb_filters
            if self.conv_block_type == "fixup":
                self.conv_blocks.append(
                    ConvBlockFixup(self.filter_width, input_filters, self.nb_filters, self.dilation)
                )
            elif self.conv_block_type == "skip":
                self.conv_blocks.append(
                    ConvBlockSkip(
                        self.window_size,
                        self.filter_width,
                        input_filters,
                        self.nb_filters,
                        self.dilation,
                        self.batch_norm,
                    )
                )
            elif self.conv_block_type == "normal":
                self.conv_blocks.append(
                    ConvBlock(
                        self.filter_width,
                        input_filters,
                        self.nb_filters,
                        self.dilation,
                        self.batch_norm,
                    )
                )
        self.conv_blocks = nn.ModuleList(self.conv_blocks)

        if self.pooling:
            if self.pool_type == "max":
                self.pool = nn.MaxPool2d((self.pool_kernel_width, 1))
            elif self.pool_type == "avg":
                self.pool = nn.AvgPool2d((self.pool_kernel_width, 1))

        if self.reduce_layer:
            self.reduce = nn.Conv2d(self.nb_filters, self.reduce_layer_output, (self.filter_width, 1))

        self.final_seq_len = self.window_size - (self.filter_width - 1) * (self.nb_conv_blocks * 2)

        if not self.no_lstm:
            self.lstm_layers = []
            for i in range(self.nb_layers_lstm):
                if i == 0:
                    if self.reduce_layer:
                        self.lstm_layers.append(
                            nn.LSTM(self.nb_channels * self.reduce_layer_output, self.nb_units_lstm)
                        )
                    else:
                        self.lstm_layers.append(nn.LSTM(self.nb_channels * self.nb_filters, self.nb_units_lstm))
                else:
                    self.lstm_layers.append(nn.LSTM(self.nb_units_lstm, self.nb_units_lstm))
            self.lstm_layers = nn.ModuleList(self.lstm_layers)

        self.dropout = nn.Dropout(self.drop_prob)
        if self.no_lstm:
            if self.reduce_layer:
                self.fc = nn.Linear(self.reduce_layer_output * self.nb_channels, self.nb_classes)
            else:
                self.fc = nn.Linear(self.nb_filters * self.nb_channels, self.nb_classes)
        else:
            self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.window_size, self.nb_channels)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        if self.pooling:
            x = self.pool(x)
            self.final_seq_len = x.shape[2]
        if self.reduce_layer:
            x = self.reduce(x)
            self.final_seq_len = x.shape[2]
        x = x.permute(0, 2, 1, 3)
        if self.reduce_layer:
            x = x.reshape(-1, self.final_seq_len, self.nb_channels * self.reduce_layer_output)
        else:
            x = x.reshape(-1, self.final_seq_len, self.nb_filters * self.nb_channels)
        if self.no_lstm:
            if self.reduce_layer:
                x = x.view(-1, self.nb_channels * self.reduce_layer_output)
            else:
                x = x.view(-1, self.nb_filters * self.nb_channels)
        else:
            for lstm_layer in self.lstm_layers:
                x, _ = lstm_layer(x)
            x = x.view(-1, self.nb_units_lstm)
        x = self.dropout(x)
        x = self.fc(x)
        out = x.view(-1, self.final_seq_len, self.nb_classes)

        return out[:, -1, :]

    def number_of_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class DeepConvLSTM_ISWC_Wrapper(ModelWrapper):
    NAME = "DeepConvLSTM_ISWC"
    display_name = "DeepConvShallowLSTM"
    color = "#f46d43"
    ARCHITECTURE = "2 conv blocks + 1-layer LSTM (ISWC 2021 config)"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(cnn=True, dense=True, recurrent=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/mariusbock/dl-for-har/blob/main/model/DeepConvLSTM.py"
    NOTES = (
        "DeepConvLSTM variant using the shallow ISWC configuration. "
        "Defaults match the paper repository's main.py network config."
    )
    INPUT_REQUIREMENTS = (
        "Expects 3D input shaped (B, L, C). With the default main.py config "
        "(2 normal conv blocks, filter_width=11), ts_len must be at least 41."
    )

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
        model_config["window_size"] = ts_len
        model_config["nb_channels"] = num_sensors
        model_config["nb_classes"] = num_classes
        self.model = DeepConvLSTM(model_config)

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] == self.num_sensors:
                return x
            raise ValueError(f"3D input must match (B, L, {self.num_sensors}); got {tuple(x.shape)}")
        raise ValueError(f"Expected 3D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        return self.model(x)


from whar_models._shared.adapter import ChannelFirstAdapter


def build_deepconv_lstm_iswc(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        DeepConvLSTM_ISWC_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
