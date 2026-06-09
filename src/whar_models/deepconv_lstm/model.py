import torch
import torch.nn as nn

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


DEFAULT_CONFIG = {
    "num_conv_layers": 4,
    "nb_filters": 64,
    "filter_width": 5,
    "num_lstm_layers": 2,
    "nb_units_lstm": 128,
}


class DeepConvLSTM(nn.Module):
    def __init__(self, input_shape, nb_classes, config):
        """
        PyTorch implementation aligned with STRCWearlab/DeepConvLSTM notebook:
        - 4 Conv2D layers (64 filters, kernel 5x1, ReLU)
        - 2 LSTM layers (128 units each)
        - Classification from the last time step
        """
        super().__init__()

        # Backward-compatible aliases from the previous local implementation.
        num_conv_layers = int(config.get("num_conv_layers", config.get("nb_conv_blocks", 4)))
        num_lstm_layers = int(config.get("num_lstm_layers", config.get("nb_layers_lstm", 2)))
        nb_filters = int(config["nb_filters"])
        filter_width = int(config["filter_width"])
        nb_units_lstm = int(config["nb_units_lstm"])

        self.nb_channels = input_shape[3]
        self.nb_filters = nb_filters
        self.nb_classes = nb_classes

        conv_layers: list[nn.Module] = []
        in_channels = input_shape[1]
        for _ in range(num_conv_layers):
            conv_layers.append(
                nn.Conv2d(in_channels, nb_filters, kernel_size=(filter_width, 1), stride=(1, 1))
            )
            conv_layers.append(nn.ReLU(inplace=True))
            in_channels = nb_filters
        self.conv = nn.Sequential(*conv_layers)

        lstm_layers: list[nn.Module] = []
        for idx in range(num_lstm_layers):
            input_size = self.nb_channels * nb_filters if idx == 0 else nb_units_lstm
            lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=nb_units_lstm, batch_first=True))
        self.lstm_layers = nn.ModuleList(lstm_layers)

        self.fc = nn.Linear(nb_units_lstm, self.nb_classes)

    def forward(self, x):
        x = self.conv(x)
        final_seq_len = x.shape[2]

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, final_seq_len, self.nb_filters * self.nb_channels)

        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)

        x = x[:, -1, :]
        x = self.fc(x)

        return x


class DeepConvLSTM_Wrapper(ModelWrapper):
    NAME = "DeepConvLSTM"
    display_name = "DeepConvLSTM"
    color = "#e08214"
    ARCHITECTURE = "4×Conv2D + 2×LSTM"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(cnn=True, dense=True, recurrent=True)
    INPUT_TYPE = "TS"
    SOURCE = "Yexu"
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

        self.model = DeepConvLSTM(
            input_shape=(1, 1, ts_len, num_sensors),
            nb_classes=num_classes,
            config=model_config,
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


def build_deepconv_lstm(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        DeepConvLSTM_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
