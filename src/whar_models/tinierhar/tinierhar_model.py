import torch
import torch.nn as nn

from whar_models._shared.wrapper import ModelWrapper


DEFAULT_CONFIG = {
    "nb_conv_blocks": 4,
    "nb_filters": 4,
    "drop_prob": 0.3,
    "nb_units_gru": 16,
    "filter_scaling_factor": 1,
}


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = (dilation * (kernel_size - 1) + 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, use_maxpool=True, shortcut=True):
        super().__init__()
        self.use_maxpool = use_maxpool
        self.shortcut = shortcut

        conv_layers = [
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if self.use_maxpool:
            conv_layers.append(nn.MaxPool2d((2, 1)))
        self.conv = nn.Sequential(*conv_layers)
        self.f_shortcut = self._create_shortcut(in_channels, out_channels)

    def _create_shortcut(self, in_channels, out_channels):
        layers = []
        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, (1, 1)))
            layers.append(nn.BatchNorm2d(out_channels))
        if self.use_maxpool:
            layers.append(nn.MaxPool2d((2, 1)))
        if not layers:
            return nn.Identity()
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.shortcut:
            return self.conv(x) + self.f_shortcut(x)
        return self.conv(x)


class TinierHARModel(nn.Module):
    # Direct adaptation of zhaxidele/TinierHAR/models/TinierHAR.py
    def __init__(self, input_shape, nb_classes, filter_scaling_factor, config):
        super().__init__()
        self.input_channels = input_shape[3]
        self.seq_length = input_shape[2]
        self.nb_conv_blocks = config["nb_conv_blocks"]
        self.nb_units_gru = config["nb_units_gru"]
        self.nb_filters = config["nb_filters"]
        self.drop_prob = config["drop_prob"]
        self.nb_classes = nb_classes

        conv_blocks = []
        conv_blocks.append(
            ConvBlock(
                1,
                self.nb_filters,
                kernel_size=5,
                dilation=1,
                use_maxpool=True,
                shortcut=True,
            )
        )
        conv_blocks.append(
            ConvBlock(
                self.nb_filters,
                2 * self.nb_filters,
                kernel_size=5,
                dilation=1,
                use_maxpool=True,
                shortcut=True,
            )
        )
        for _ in range(self.nb_conv_blocks):
            conv_blocks.append(
                ConvBlock(
                    2 * self.nb_filters,
                    2 * self.nb_filters,
                    kernel_size=5,
                    dilation=1,
                    use_maxpool=False,
                    shortcut=True,
                )
            )
        self.conv_blocks = nn.Sequential(*conv_blocks)

        with torch.no_grad():
            dummy = torch.randn(1, 1, self.seq_length, self.input_channels)
            out = self.conv_blocks(dummy)
            gru_input_dim = out.size(1) * out.size(3)

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=self.nb_units_gru,
            bidirectional=True,
            batch_first=True,
        )

        self.attention = nn.Linear(2 * self.nb_units_gru, 1)
        self.classifier = nn.Sequential(nn.Linear(2 * self.nb_units_gru, self.nb_classes))
        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        x = self.conv_blocks(x)
        bsz, channels, tlen, cin = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, tlen, -1)
        x = self.dropout(x)
        x, _ = self.gru(x)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)
        return self.classifier(x)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinierHARWrapper(ModelWrapper):
    NAME = "TinierHAR"
    ARCHITECTURE = "Depthwise-separable Conv2D blocks + BiGRU + attention"
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/zhaxidele/TinierHAR"
    NOTES = ""

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
                x = x.unsqueeze(1)
            elif x.shape[1] == self.num_sensors:
                x = x.transpose(1, 2).unsqueeze(1)
            else:
                raise ValueError(
                    f"3D input must match (B, L, {self.num_sensors}) or (B, {self.num_sensors}, L); got {tuple(x.shape)}"
                )
        elif x.ndim == 4:
            if x.shape[1] != 1 or x.shape[3] != self.num_sensors:
                raise ValueError(
                    f"4D input must match (B, 1, L, {self.num_sensors}); got {tuple(x.shape)}"
                )
        else:
            raise ValueError(
                f"Expected 3D or 4D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}"
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        return self.model(x)
