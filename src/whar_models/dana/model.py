from __future__ import annotations

import math

import torch
import torch.nn as nn

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


class DimensionAdaptivePooling(nn.Module):
    def __init__(self, pooling_parameters: tuple[int, int], for_rnn: bool = False, operation: str = "max") -> None:
        super().__init__()
        self.pool_list = tuple(int(value) for value in pooling_parameters)
        self.for_rnn = for_rnn
        self.W = self.pool_list[0]
        self.H = self.pool_list[1]
        if operation == "max":
            self.operation = torch.amax
        elif operation == "avg":
            self.operation = torch.mean
        else:
            raise ValueError(f"Unsupported DAP operation: {operation}")

    def forward(self, xp: torch.Tensor) -> torch.Tensor:
        input_shape = xp.shape
        # Use explicit ints for TorchScript tracing; xp.shape entries can be traced tensor-like values.
        wp = int(xp.size(1))
        hp = int(xp.size(2))
        xpp = xp

        A = max(math.ceil((self.H - hp) / 3), 0)
        for _ in range(A):
            xpp = torch.cat([xpp, xp], dim=2)
        xpp = xpp[:, :wp, : max(hp, self.H), :]

        p_w = wp / self.W
        p_h = hp / self.H

        pooled_regions: list[torch.Tensor] = []
        for iw in range(self.W):
            for ih in range(self.H):
                r1 = int(round(iw * p_w))
                r2 = int(round((iw + 1) * p_w))
                if A == 0:
                    c1 = int(round(ih * p_h))
                    c2 = int(round((ih + 1) * p_h))
                else:
                    stride_h = math.floor((A + 1) * p_h)
                    c1 = int(round(ih * stride_h))
                    c2 = int(round((ih + 1) * stride_h))

                if r2 <= r1:
                    r2 = min(wp, r1 + 1)
                if c2 <= c1:
                    c2 = min(xpp.shape[2], c1 + 1)

                region = xpp[:, r1:r2, c1:c2, :]
                pooled_regions.append(self.operation(region, dim=(1, 2)))

        Zp = torch.cat(pooled_regions, dim=-1)
        if self.for_rnn:
            return Zp.reshape(input_shape[0], self.W, self.H * input_shape[3])
        return Zp.reshape(input_shape[0], self.W * self.H * input_shape[3])


class DimensionAdaptivePoolingForSensors(DimensionAdaptivePooling):
    def __init__(self, pooling_parameters: tuple[int, int], for_rnn: bool = False, operation: str = "max") -> None:
        super().__init__(pooling_parameters=pooling_parameters, for_rnn=for_rnn, operation=operation)


class Ordonez2016DeepWithDAP(nn.Module):
    def __init__(self, inp_shape: tuple[int | None, int | None, int], out_shape: int, pool_list: tuple[int, int]) -> None:
        super().__init__()
        _ = inp_shape
        nb_filters = 64
        drp_out_dns = 0.5
        nb_dense = 128

        self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=(5, 1), stride=(1, 1), padding="same")
        self.conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=(5, 1), stride=(1, 1), padding="same")
        self.conv3 = nn.Conv2d(nb_filters, nb_filters, kernel_size=(5, 1), stride=(1, 1), padding="same")
        self.conv4 = nn.Conv2d(nb_filters, nb_filters, kernel_size=(5, 1), stride=(1, 1), padding="same")
        self.relu = nn.ReLU()
        self.dap = DimensionAdaptivePoolingForSensors(pool_list, operation="max", for_rnn=True)
        self.lstm_1 = nn.LSTM(pool_list[1] * nb_filters, nb_dense, batch_first=True)
        self.dot_1 = nn.Dropout(drp_out_dns)
        self.lstm_2 = nn.LSTM(nb_dense, nb_dense, batch_first=True)
        self.dot_2 = nn.Dropout(drp_out_dns)
        self.act_smx = nn.Linear(nb_dense, out_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upstream Keras code uses channels-last tensors shaped (B, T, C, 1).
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.permute(0, 2, 3, 1)
        x = self.dap(x)
        act, _ = self.lstm_1(x)
        act = self.dot_1(act)
        act, _ = self.lstm_2(act)
        act = act[:, -1, :]
        act = self.dot_2(act)
        return self.act_smx(act)


DEFAULT_CONFIG = {
    "pool_list": None,
}


class DANA_Wrapper(ModelWrapper):
    NAME = "DANA"
    display_name = "DANA"
    color = "#d73027"
    ARCHITECTURE = "DeepConvLSTM + dimension-adaptive pooling"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(cnn=True, dense=True, recurrent=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/mmalekzadeh/dana"
    NOTES = (
        "Direct PyTorch translation of dana.models.Ordonez2016DeepWithDAP and dana.dap.DimensionAdaptivePoolingForSensors. "
        "No architecture logic was intentionally changed; only wrapper integration and TensorFlow-to-PyTorch translation were added."
    )
    INPUT_REQUIREMENTS = (
        "Expects 3D input shaped (B, L, C). Internally translated to channels-last (B, L, C, 1) "
        "to match the upstream Keras model."
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

        pool_list = model_config.get("pool_list")
        if pool_list is None:
            pool_list = (8, num_sensors)
        pool_list = (int(pool_list[0]), int(pool_list[1]))

        self.model = Ordonez2016DeepWithDAP(
            inp_shape=(None, None, 1),
            out_shape=num_classes,
            pool_list=pool_list,
        )

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] != self.num_sensors:
                raise ValueError(f"3D input must match (B, L, {self.num_sensors}); got {tuple(x.shape)}")
            return x.unsqueeze(-1)
        if x.ndim == 4:
            if x.shape[3] != 1 or x.shape[2] != self.num_sensors:
                raise ValueError(f"4D input must match (B, L, {self.num_sensors}, 1); got {tuple(x.shape)}")
            return x
        raise ValueError(f"Expected 3D or 4D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        return self.model(x)


from whar_models._shared.adapter import ChannelFirstAdapter


def build_dana(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        DANA_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
