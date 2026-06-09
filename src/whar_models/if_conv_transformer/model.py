"""
Adapted from:
https://github.com/crocodilegogogo/IF-ConvTransformer-UbiComp2022
Reference file:
src/classifiers/If_ConvTransformer_torch.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


DEFAULT_CONFIG = {
    "input_2dfeature_channel": 1,
    "input_channel": 9,
    "feature_channel": 64,
    "kernel_size": 5,
    "kernel_size_grav": 3,
    "scale_num": 2,
    "feature_channel_out": 128,
    "multiheads": 1,
    "drop_rate": 0.2,
}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 128) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + Variable(self.pe[:, :, : x.size(2)], requires_grad=False)
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, k: int, heads: int = 8, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.k = k
        self.heads = heads
        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)
        self.dropout_attention = nn.Dropout(drop_rate)
        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tlen, kdim = x.size()
        heads = self.heads
        queries = self.toqueries(x).view(bsz, tlen, heads, kdim)
        keys = self.tokeys(x).view(bsz, tlen, heads, kdim)
        values = self.tovalues(x).view(bsz, tlen, heads, kdim)

        queries = queries.transpose(1, 2).contiguous().view(bsz * heads, tlen, kdim)
        keys = keys.transpose(1, 2).contiguous().view(bsz * heads, tlen, kdim)
        values = values.transpose(1, 2).contiguous().view(bsz * heads, tlen, kdim)

        queries = queries / (kdim ** (1 / 4))
        keys = keys / (kdim ** (1 / 4))
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)
        dot = self.dropout_attention(dot)
        out = torch.bmm(dot, values).view(bsz, heads, tlen, kdim)
        out = out.transpose(1, 2).contiguous().view(bsz, tlen, heads * kdim)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k: int, heads: int, drop_rate: float) -> None:
        super().__init__()
        self.attention = SelfAttention(k, heads=heads, drop_rate=drop_rate)
        self.norm1 = nn.BatchNorm1d(k)
        self.mlp = nn.Sequential(nn.Linear(k, 4 * k), nn.ReLU(), nn.Linear(4 * k, k))
        self.norm2 = nn.BatchNorm1d(k)
        self.dropout_forward = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x) + x
        x = self.norm1(attended.permute(0, 2, 1)).permute(0, 2, 1)
        feedforward = self.mlp(x) + x
        return self.dropout_forward(self.norm2(feedforward.permute(0, 2, 1)).permute(0, 2, 1))


class Chomp2d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :, : -self.chomp_size].contiguous()


class IMUFusionBlock(nn.Module):
    def __init__(
        self,
        input_2dfeature_channel: int,
        input_channel: int,
        feature_channel: int,
        kernel_size_grav: int,
        scale_num: int,
    ) -> None:
        super().__init__()
        self.scale_num = scale_num
        self.input_channel = input_channel
        if self.input_channel % 3 != 0:
            raise ValueError(
                f"IFConvTransformer expects input_channel divisible by 3 (grav/gyro/acc groups), got {self.input_channel}"
            )
        self.channels_per_modality = self.input_channel // 3
        self.tcn_grav_convs = nn.ModuleList()
        self.tcn_gyro_convs = nn.ModuleList()
        self.tcn_acc_convs = nn.ModuleList()

        for i in range(scale_num):
            dilation_num_grav = i + 1
            padding_grav = (kernel_size_grav - 1) * dilation_num_grav
            kernel_size_gyro = padding_grav
            kernel_size_acc = padding_grav + 1

            tcn_grav = nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        input_2dfeature_channel,
                        feature_channel,
                        (1, kernel_size_grav),
                        1,
                        (0, padding_grav),
                        dilation=dilation_num_grav,
                    )
                ),
                Chomp2d(padding_grav),
                nn.ReLU(),
            )

            if kernel_size_gyro == 1:
                tcn_gyro = nn.Sequential(
                    weight_norm(nn.Conv2d(input_2dfeature_channel, feature_channel, (1, 1), 1, (0, 0), dilation=1)),
                    nn.ReLU(),
                )
            else:
                gyro_pad = kernel_size_gyro - 1
                tcn_gyro = nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            input_2dfeature_channel,
                            feature_channel,
                            (1, kernel_size_gyro),
                            1,
                            (0, gyro_pad),
                            dilation=1,
                        )
                    ),
                    Chomp2d(gyro_pad),
                    nn.ReLU(),
                )

            acc_pad = kernel_size_acc - 1
            tcn_acc = nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        input_2dfeature_channel,
                        feature_channel,
                        (1, kernel_size_acc),
                        1,
                        (0, acc_pad),
                        dilation=1,
                    )
                ),
                Chomp2d(acc_pad),
                nn.ReLU(),
            )

            self.tcn_grav_convs.append(tcn_grav)
            self.tcn_gyro_convs.append(tcn_gyro)
            self.tcn_acc_convs.append(tcn_acc)

        self.attention = nn.Sequential(
            nn.Linear(self.channels_per_modality * feature_channel, 1),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.channels_per_modality
        x_grav = x[:, :, 0:c, :]
        x_gyro = x[:, :, c : 2 * c, :]
        x_acc = x[:, :, 2 * c : 3 * c, :]

        out_attitude = None
        out_dynamic = None
        for i in range(self.scale_num):
            out_grav = self.tcn_grav_convs[i](x_grav).unsqueeze(4)
            out_gyro = self.tcn_gyro_convs[i](x_gyro).unsqueeze(4)
            out_acc = self.tcn_acc_convs[i](x_acc)

            if out_attitude is None:
                out_attitude = torch.cat([out_grav, out_gyro], dim=4)
                out_dynamic = out_acc
            else:
                out_attitude = torch.cat([out_attitude, out_grav], dim=4)
                out_attitude = torch.cat([out_attitude, out_gyro], dim=4)
                out_dynamic = torch.cat([out_dynamic, out_acc], dim=2)

        out_attitude = out_attitude.permute(0, 3, 4, 2, 1)
        out_attitude = out_attitude.reshape(out_attitude.shape[0], out_attitude.shape[1], out_attitude.shape[2], -1)
        sensor_attn = self.attention(out_attitude).squeeze(3)
        sensor_attn = F.softmax(sensor_attn, dim=2).unsqueeze(-1)
        out_attitude = sensor_attn * out_attitude

        norm_num = torch.mean(sensor_attn.squeeze(-1), dim=1)
        norm_num = torch.pow(norm_num, 2)
        norm_num = torch.sqrt(torch.sum(norm_num, dim=1))
        norm_num = (pow(self.scale_num, 0.5) / norm_num).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        out_attitude = out_attitude * norm_num

        out_attitude = out_attitude.reshape(
            out_attitude.shape[0],
            out_attitude.shape[1],
            out_attitude.shape[2],
            c,
            -1,
        )
        out_attitude = out_attitude.reshape(
            out_attitude.shape[0],
            out_attitude.shape[1],
            out_attitude.shape[2] * c,
            -1,
        )
        out_attitude = out_attitude.permute(0, 3, 2, 1)

        split_attitude = torch.split(out_attitude, 2 * c, dim=2)
        all_attitude = None
        for j, per_scale in enumerate(split_attitude):
            per_scale_attitude = torch.split(per_scale, c, dim=2)
            per_attitude = None
            for part in per_scale_attitude:
                per_attitude = part if per_attitude is None else per_attitude + part
            all_attitude = per_attitude if j == 0 else torch.cat([all_attitude, per_attitude], dim=2)

        out = torch.cat([all_attitude, out_dynamic], dim=2)
        return out, sensor_attn


class IFConvTransformer(nn.Module):
    def __init__(
        self,
        input_2dfeature_channel: int,
        input_channel: int,
        feature_channel: int,
        kernel_size: int,
        kernel_size_grav: int,
        scale_num: int,
        feature_channel_out: int,
        multiheads: int,
        drop_rate: float,
        data_length: int,
        num_class: int,
    ) -> None:
        super().__init__()
        self.imu_fusion_block = IMUFusionBlock(
            input_2dfeature_channel=input_2dfeature_channel,
            input_channel=input_channel,
            feature_channel=feature_channel,
            kernel_size_grav=kernel_size_grav,
            scale_num=scale_num,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )

        reduced_channel = input_channel // 3
        self.transition = nn.Sequential(
            nn.Conv1d(feature_channel * (input_channel - reduced_channel) * scale_num, feature_channel_out, 1, 1),
            nn.BatchNorm1d(feature_channel_out),
            nn.ReLU(),
        )
        self.position_encode = PositionalEncoding(feature_channel_out, drop_rate, data_length)
        self.transformer_block1 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        self.transformer_block2 = TransformerBlock(feature_channel_out, multiheads, drop_rate)
        self.global_ave_pooling = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(feature_channel_out, num_class)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 1, C, L)
        batch_size = x.shape[0]
        data_length = x.shape[-1]
        x, out_attn = self.imu_fusion_block(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(batch_size, -1, data_length)
        x = self.transition(x)
        x = self.position_encode(x)
        x = x.permute(0, 2, 1)
        x = self.transformer_block1(x)
        x = self.transformer_block2(x)
        x = x.permute(0, 2, 1)
        x = self.global_ave_pooling(x).squeeze(-1)
        output = self.linear(x)
        return output, out_attn


class IFConvTransformerWrapper(ModelWrapper):
    NAME = "IF-ConvTransformer"
    display_name = "IF-ConvTransformer"
    color = "#f15bb5"
    ARCHITECTURE = "Multi-scale TCN + Transformer"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(attention=True, cnn=True, dense=True, transformer=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/crocodilegogogo/IF-ConvTransformer-UbiComp2022"
    NOTES = (
        "PyTorch port of the original IF-ConvTransformer model body with wrapper-level adaptations for the benchmark API. "
        "This implementation is not fully faithful to the original source because it generalizes the fixed 9-channel "
        "grav/gyro/acc layout to arbitrary divisible-by-3 channel counts, and may apply a sensor_adapter projection when "
        "benchmark num_sensors differs from the configured input_channel."
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
        explicit_input_channel = False
        if config is not None and "input_channel" in config:
            explicit_input_channel = True
        if "input_channel" in kwargs:
            explicit_input_channel = True

        model_config = dict(DEFAULT_CONFIG)
        if config is not None:
            model_config.update(config)
        model_config.update(kwargs)

        if explicit_input_channel:
            self.model_input_channels = int(model_config["input_channel"])
        else:
            self.model_input_channels = num_sensors
            model_config["input_channel"] = num_sensors

        self.sensor_adapter = None
        if num_sensors != self.model_input_channels:
            self.sensor_adapter = nn.Linear(num_sensors, self.model_input_channels)

        self.model = IFConvTransformer(
            input_2dfeature_channel=int(model_config["input_2dfeature_channel"]),
            input_channel=self.model_input_channels,
            feature_channel=int(model_config["feature_channel"]),
            kernel_size=int(model_config["kernel_size"]),
            kernel_size_grav=int(model_config["kernel_size_grav"]),
            scale_num=int(model_config["scale_num"]),
            feature_channel_out=int(model_config["feature_channel_out"]),
            multiheads=int(model_config["multiheads"]),
            drop_rate=float(model_config["drop_rate"]),
            data_length=ts_len,
            num_class=num_classes,
        )

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] == self.num_sensors:
                x = x.unsqueeze(1)
            else:
                raise ValueError(f"3D input must match (B, L, {self.num_sensors}); got {tuple(x.shape)}")
        else:
            raise ValueError(f"Expected 3D input tensor, got {x.ndim}D with shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)  # (B,1,L,C)
        if self.sensor_adapter is not None:
            x = self.sensor_adapter(x.squeeze(1))  # (B,L,C_model)
            x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)  # (B,1,C,L)
        logits, _ = self.model(x)
        return logits


from whar_models._shared.adapter import ChannelFirstAdapter


def build_if_conv_transformer(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        IFConvTransformerWrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
