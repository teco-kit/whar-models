from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


class CNN_acc(nn.Module):
    def __init__(self):
        super(CNN_acc, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (1, 3), padding="same")
        self.conv2 = nn.Conv2d(32, 32, (1, 3), padding="same")
        self.conv3 = nn.Conv2d(32, 64, (1, 3), padding="same")
        self.conv4 = nn.Conv2d(64, 64, (1, 3), padding="same")
        self.fc = nn.LazyLinear(128)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.xavier_normal_(self.conv4.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), (1, 2)))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), (1, 2)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


class CNN_gyro(nn.Module):
    def __init__(self):
        super(CNN_gyro, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (1, 3), padding="same")
        self.conv2 = nn.Conv2d(32, 32, (1, 3), padding="same")
        self.conv3 = nn.Conv2d(32, 64, (1, 3), padding="same")
        self.conv4 = nn.Conv2d(64, 64, (1, 3), padding="same")
        self.fc = nn.LazyLinear(128)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.xavier_normal_(self.conv4.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), (1, 2)))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), (1, 2)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


class Attention2(nn.Module):
    def __init__(self):
        super(Attention2, self).__init__()
        self.v = torch.rand(64, 1)
        self.v = torch.nn.Parameter(torch.nn.init.xavier_uniform_(self.v))
        self.new_linear = torch.nn.Linear(64, 64)

    def forward(self, x):
        temp_o = self.new_linear(x)
        o = torch.tanh(temp_o)
        temp_w = o @ self.v
        w = F.softmax(temp_w, dim=1)
        context = torch.sum(w * temp_o, dim=1)

        return context


class Attnsense(nn.Module):
    def __init__(self, num_channels: int, num_classes: int = 8):
        super(Attnsense, self).__init__()
        self.num_channels = int(num_channels)
        self.conv_acc = CNN_acc()
        # The wrapper generalizes the original two-branch design by encoding every
        # input channel with the same CNN backbone, so the second branch is unused.
        # Keeping an unused LazyLinear-backed module breaks export flows.
        self.conv_gyro = nn.Identity()
        self.attn1 = torch.nn.MultiheadAttention(128, 1, batch_first=True)
        self.attn2 = Attention2()
        self.gru = nn.GRU(128, 64, num_layers=2, batch_first=True)
        self.fc_gru = nn.Linear(64, num_classes)
        torch.nn.init.xavier_normal_(self.fc_gru.weight)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, 5, 13, 10), got {tuple(x.shape)}")

        batch_size, num_channels, num_steps, spec_h, spec_w = x.shape
        if num_channels != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {num_channels}")

        # Encode each channel independently at each temporal step.
        channel_outs = []
        for channel_index in range(self.num_channels):
            channel_input = x[:, channel_index, :, :, :].reshape(batch_size * num_steps, 1, spec_h, spec_w)
            channel_out = self.conv_acc(channel_input).reshape(batch_size, num_steps, 128)
            channel_outs.append(channel_out)

        # Attention across channels for each temporal step.
        zipped = torch.stack(channel_outs, dim=2)  # (B, T, C, 128)
        attn_in = zipped.reshape(batch_size * num_steps, num_channels, 128)
        attn1_out = self.attn1(attn_in, attn_in, attn_in)[0].mean(dim=1)
        attn1_out = attn1_out.reshape(batch_size, num_steps, 128)

        # Temporal modeling across the 5 spectrogram steps.
        out_gru = self.gru(attn1_out)[0]
        out_attn2 = self.attn2(out_gru)
        x = self.fc_gru(out_attn2)
        return x


class AttenSenseWrapper(ModelWrapper):
    NAME = "AttenSense"
    display_name = "AttenSense"
    color = "#ff4fa3"
    ARCHITECTURE = "spectrogram CNN + channel attention + GRU + temporal attention"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(
        attention=True,
        cnn=True,
        dense=True,
        recurrent=True,
        spectral=True,
    )
    INPUT_TYPE = "TS->Spec"
    SOURCE = "User-provided Colab notebook code"
    NOTES = (
        "The wrapped AttenSense model keeps the original CNN, attention, and GRU stack as close as possible. "
        "The original code assumed exactly two modality groups (acc and gyro). "
        "This minimal adaptation instead treats each input channel independently, builds one spectrogram per channel, "
        "encodes every channel with the original CNN backbone, and applies the existing attention stack across channels. "
        "The first attention output is averaged across channels to produce one feature vector per temporal step, "
        "leaving the downstream GRU and temporal attention logic unchanged. "
        "The CNN projection layer now uses LazyLinear so the original fixed 2048 flatten size is no longer required. "
        "This removes the need for the original explicit Xavier init on that projection because LazyLinear materializes "
        "its weights only on the first forward pass. "
        "The classification head is resized to match the requested num_classes."
    )
    INPUT_REQUIREMENTS = (
        "Accepts any num_sensors >= 1. Raw inputs must be shaped as (B, L, C). "
        "Precomputed spectrogram inputs must be shaped as (B, C, 5, 13, 10)."
    )

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        ts_len: int = 128,
        raw_window_len: int = 800,
        spec_window_len: int = 200,
        spec_hop_len: int = 150,
        n_fft: int = 25,
        noverlap: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(num_sensors=num_sensors, num_classes=num_classes)
        _ = ts_len, kwargs
        if num_sensors < 1:
            raise ValueError(f"AttenSenseWrapper requires at least 1 sensor; got {num_sensors}")

        self.raw_window_len = raw_window_len
        self.spec_window_len = spec_window_len
        self.spec_hop_len = spec_hop_len
        self.n_fft = n_fft
        self.noverlap = noverlap
        self.stft_hop_len = self.n_fft - self.noverlap
        self.num_spec_steps = 5

        expected_raw_len = self.spec_window_len + self.spec_hop_len * (self.num_spec_steps - 1)
        if self.raw_window_len != expected_raw_len:
            raise ValueError(
                f"raw_window_len must equal {expected_raw_len} to match the copied AttenSense model; "
                f"got {self.raw_window_len}"
            )

        self.model = Attnsense(num_channels=num_sensors, num_classes=num_classes)
        self.register_buffer("spec_window", torch.hamming_window(self.n_fft, periodic=False))

    def _to_sequence_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] == self.num_sensors:
                return x
            raise ValueError(f"3D input must match (B, L, {self.num_sensors}); got {tuple(x.shape)}")
        raise ValueError(f"Expected 3D raw input, or 5D spectrogram input; got {x.ndim}D input with shape {tuple(x.shape)}")

    def _resample_temporal(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.shape[1] == target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.stft_hop_len,
            win_length=self.n_fft,
            window=self.spec_window,
            center=False,
            return_complex=True,
        )
        return stft.abs().pow(2)

    def _build_channel_sequences(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resample_temporal(x, self.raw_window_len)
        starts = range(0, self.raw_window_len - self.spec_window_len + 1, self.spec_hop_len)
        segments = [x[:, start : start + self.spec_window_len, :] for start in starts]
        stacked = torch.stack(segments, dim=1)

        batch_size, num_steps, _, num_channels = stacked.shape
        per_channel = stacked.permute(0, 3, 1, 2).reshape(batch_size * num_channels * num_steps, self.spec_window_len)
        specs = self._spectrogram(per_channel)
        return specs.reshape(batch_size, num_channels, num_steps, specs.shape[-2], specs.shape[-1])

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            expected = (self.num_sensors, self.num_spec_steps, 13, 10)
            if tuple(x.shape[1:]) != expected:
                raise ValueError(
                    f"5D spectrogram input must match (B, {expected[0]}, {expected[1]}, {expected[2]}, {expected[3]}); "
                    f"got {tuple(x.shape)}"
                )
            return x

        x = self._to_sequence_input(x)
        return self._build_channel_sequences(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        return self.model(x)


from whar_models._shared.adapter import ChannelFirstAdapter


def build_attensense(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        AttenSenseWrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
