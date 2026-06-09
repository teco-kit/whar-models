"""
Reference-inspired DeepSense port based on:
https://github.com/yscacaca/DeepSense/blob/1979ba414d3cfd0a84a5247e2713540e8c225ec7/deepSense_HHAR_tf.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


DEFAULT_CONFIG = {
    "spectral_samples": 10,   # SEPCTURAL_SAMPLES
    "window_count": 20,       # WIDE
    "conv_len": 3,            # CONV_LEN
    "conv_len_inte": 3,       # CONV_LEN_INTE
    "conv_len_last": 3,       # CONV_LEN_LAST
    "conv_num": 64,           # CONV_NUM
    "conv_num2": 64,          # CONV_NUM2
    "conv_merge_len": 8,      # CONV_MERGE_LEN
    "conv_merge_len2": 6,     # CONV_MERGE_LEN2
    "conv_merge_len3": 4,     # CONV_MERGE_LEN3
    "inter_dim": 120,         # INTER_DIM
    "conv_keep_prob": 0.8,    # CONV_KEEP_PROB
    "gru_output_keep_prob": 0.5,  # DropoutWrapper output_keep_prob
}


def compute_spectral_features(
    x: torch.Tensor,
    num_sensors: int,
    window_count: int,
    spectral_samples: int,
) -> torch.Tensor:
    # (B, 1, L, C) -> (B, W, C*2*S)
    x = x.squeeze(1).transpose(1, 2)
    target_len = window_count * spectral_samples
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    x = x.view(x.shape[0], num_sensors, window_count, spectral_samples)
    x = x.permute(0, 2, 1, 3)
    fft = torch.fft.fft(x, dim=-1)
    spectral = torch.cat([fft.real, fft.imag], dim=-1)
    return spectral.reshape(spectral.shape[0], window_count, -1)


def _same_pad_3d(x: torch.Tensor, kernel_size: tuple[int, int, int], stride: tuple[int, int, int]) -> torch.Tensor:
    # Applies TF-like SAME padding for Conv3d.
    pads: list[int] = []
    for dim, (k, s) in zip((4, 3, 2), zip(reversed(kernel_size), reversed(stride))):
        in_size = x.shape[dim]
        out_size = (in_size + s - 1) // s
        pad_needed = max(0, (out_size - 1) * s + k - in_size)
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before
        pads.extend([pad_before, pad_after])
    return F.pad(x, pads)


class BranchConv(nn.Module):
    # Matches acc_* and gyro_* conv stacks from the reference TF code.
    def __init__(
        self,
        conv_num: int,
        kernel1_width: int,
        stride1_width: int,
        conv_len_inte: int,
        conv_len_last: int,
        conv_dropout_p: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv_num, kernel_size=(1, kernel1_width), stride=(1, stride1_width))
        self.bn1 = nn.BatchNorm2d(conv_num)
        self.drop1 = nn.Dropout2d(conv_dropout_p)

        self.conv2 = nn.Conv2d(conv_num, conv_num, kernel_size=(1, conv_len_inte), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(conv_num)
        self.drop2 = nn.Dropout2d(conv_dropout_p)

        self.conv3 = nn.Conv2d(conv_num, conv_num, kernel_size=(1, conv_len_last), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(conv_num)
        self.drop3 = nn.Dropout2d(conv_dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        return x


class SamePadConv3dBnReluDrop(nn.Module):
    # Matches sensor_conv* layers in the reference TF code (NDHWC + SAME).
    def __init__(self, channels: int, kernel_size: tuple[int, int, int], dropout_p: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = (1, 1, 1)
        self.conv = nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=self.stride, padding=0)
        self.bn = nn.BatchNorm3d(channels)
        self.drop = nn.Dropout3d(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _same_pad_3d(x, self.kernel_size, self.stride)
        x = F.relu(self.bn(self.conv(x)))
        x = self.drop(x)
        return x


class DeepSense(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int, int], nb_classes: int, config: dict) -> None:
        super().__init__()
        self.num_sensors = int(input_shape[3])
        if self.num_sensors % 2 != 0:
            raise ValueError(f"DeepSense expects an even sensor count for acc/gyro split; got {self.num_sensors}")

        self.nb_classes = nb_classes
        self.spectral_samples = int(config["spectral_samples"])
        self.window_count = int(config["window_count"])
        self.conv_len = int(config["conv_len"])
        self.conv_len_inte = int(config["conv_len_inte"])
        self.conv_len_last = int(config["conv_len_last"])
        self.conv_num = int(config["conv_num"])
        self.conv_num2 = int(config["conv_num2"])
        self.conv_merge_len = int(config["conv_merge_len"])
        self.conv_merge_len2 = int(config["conv_merge_len2"])
        self.conv_merge_len3 = int(config["conv_merge_len3"])
        self.inter_dim = int(config["inter_dim"])
        conv_dropout_p = 1.0 - float(config["conv_keep_prob"])
        gru_dropout_p = 1.0 - float(config["gru_output_keep_prob"])

        # For HHAR-like 6 channels (3 acc + 3 gyro), this becomes kernel=18, stride=6.
        branch_stride = self.num_sensors
        branch_kernel = self.num_sensors * self.conv_len
        self.acc_branch = BranchConv(
            conv_num=self.conv_num,
            kernel1_width=branch_kernel,
            stride1_width=branch_stride,
            conv_len_inte=self.conv_len_inte,
            conv_len_last=self.conv_len_last,
            conv_dropout_p=conv_dropout_p,
        )
        self.gyro_branch = BranchConv(
            conv_num=self.conv_num,
            kernel1_width=branch_kernel,
            stride1_width=branch_stride,
            conv_len_inte=self.conv_len_inte,
            conv_len_last=self.conv_len_last,
            conv_dropout_p=conv_dropout_p,
        )

        self.sensor_conv1 = SamePadConv3dBnReluDrop(
            channels=self.conv_num2,
            kernel_size=(1, 2, self.conv_merge_len),
            dropout_p=conv_dropout_p,
        )
        self.sensor_conv2 = SamePadConv3dBnReluDrop(
            channels=self.conv_num2,
            kernel_size=(1, 2, self.conv_merge_len2),
            dropout_p=conv_dropout_p,
        )
        self.sensor_conv3 = SamePadConv3dBnReluDrop(
            channels=self.conv_num2,
            kernel_size=(1, 2, self.conv_merge_len3),
            dropout_p=conv_dropout_p,
        )

        fusion_dim = self._infer_fusion_dim()
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=self.inter_dim,
            num_layers=2,
            batch_first=True,
            dropout=gru_dropout_p,
        )
        self.fc = nn.Linear(self.inter_dim, self.nb_classes)

    def _branch_forward(self, x_half: torch.Tensor, branch: BranchConv) -> torch.Tensor:
        # x_half: (B, W, F_half) -> (B, W, 1, width, conv_num)
        x_half = x_half.unsqueeze(1)  # (B, 1, W, F_half)
        out = branch(x_half)          # (B, conv_num, W, width)
        out = out.permute(0, 2, 3, 1).unsqueeze(2)  # (B, W, 1, width, conv_num)
        return out

    def _sensor_fusion(self, acc: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        # concat over sensor-branch dimension -> (B, W, 2, width, conv_num)
        x = torch.cat([acc, gyro], dim=2)
        # to Conv3d format: (B, C, D, H, W) = (B, conv_num, W, 2, width)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.sensor_conv1(x)
        x = self.sensor_conv2(x)
        x = self.sensor_conv3(x)
        # flatten per time step -> (B, W, feat)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def _infer_fusion_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.window_count, self.num_sensors * self.spectral_samples * 2)
            feats = dummy
            acc_in, gyro_in = torch.chunk(feats, 2, dim=2)
            acc = self._branch_forward(acc_in, self.acc_branch)
            gyro = self._branch_forward(gyro_in, self.gyro_branch)
            fused = self._sensor_fusion(acc, gyro)
        return int(fused.shape[-1])

    def forward_spectral(self, feats: torch.Tensor) -> torch.Tensor:
        # feats shape: (B, window_count, feature_dim)
        used = torch.sign(torch.amax(feats.abs(), dim=2))  # (B, W)
        lengths = used.sum(dim=1).long()
        lengths = torch.clamp(lengths, min=1, max=self.window_count)

        acc_in, gyro_in = torch.chunk(feats, 2, dim=2)
        acc = self._branch_forward(acc_in, self.acc_branch)
        gyro = self._branch_forward(gyro_in, self.gyro_branch)
        fused = self._sensor_fusion(acc, gyro)

        packed = pack_padded_sequence(fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=self.window_count)

        mask = used.unsqueeze(-1)
        avg_num = mask.sum(dim=1).clamp_min(1.0)
        avg_cell_out = (outputs * mask).sum(dim=1) / avg_num
        return self.fc(avg_cell_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"DeepSense.forward expects spectral features with shape (B, {self.window_count}, F); "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.window_count:
            raise ValueError(
                f"Expected spectral sequence length {self.window_count}, got {x.shape[1]}"
            )
        return self.forward_spectral(x)


class DeepSenseWrapper(ModelWrapper):
    NAME = "DeepSense"
    display_name = "DeepSense"
    color = "#f94144"
    ARCHITECTURE = "FFT + dual-branch Conv2D + sensor-fusion Conv3D + GRU"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(cnn=True, dense=True, recurrent=True, spectral=True)
    INPUT_TYPE = "TS→FFT"
    SOURCE = "https://github.com/yscacaca/DeepSense"
    NOTES = "Requires even num_sensors for acc/gyro split."
    INPUT_REQUIREMENTS = (
        "num_sensors must be even. Features are split into two equal halves "
        "and processed as acc and gyro branches."
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

        self.model = DeepSense(
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
        spectral = compute_spectral_features(
            x=x,
            num_sensors=self.num_sensors,
            window_count=self.model.window_count,
            spectral_samples=self.model.spectral_samples,
        )
        return self.model.forward_spectral(spectral)


from whar_models._shared.adapter import ChannelFirstAdapter


def build_deepsense(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        DeepSenseWrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
