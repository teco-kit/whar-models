from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


def extract_blocks_with_overlap_torch(
    input_segs: torch.Tensor, segment_length: int, overlap_ratio: float = 0.5
) -> torch.Tensor:
    batch_size, features, total_steps, num_channels = input_segs.size()
    if features != 1:
        raise ValueError(f"Expected input with a singleton feature axis, got shape {tuple(input_segs.shape)}")
    if segment_length <= 0:
        raise ValueError(f"segment_length must be positive, got {segment_length}")
    if total_steps < segment_length:
        raise ValueError(
            f"segment_length={segment_length} must not exceed the time dimension {total_steps}"
        )
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")

    overlap = int(segment_length * overlap_ratio)
    step = max(1, segment_length - overlap)
    starts = list(range(0, total_steps - segment_length + 1, step))
    if starts[-1] != total_steps - segment_length:
        starts.append(total_steps - segment_length)

    blocks: list[torch.Tensor] = []
    for start in starts:
        blocks.append(input_segs[:, :, start : start + segment_length, :])
    return torch.cat(blocks, dim=-1)


class EmbeddingBlock(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        segment_length: int,
        feature_dim: int,
        temporal_flag: bool = True,
        fft_flag: bool = True,
        share_flag: bool = False,
        fuse_early: bool = False,
        overlap_ratio: float = 0.5,
    ) -> None:
        super().__init__()

        _, features, total_steps, num_channels = input_shape
        if features != 1:
            raise ValueError(f"Expected input shape (B, 1, T, C), got {input_shape}")
        if fuse_early and num_channels % 3 != 0:
            raise ValueError(
                f"fuse_early=True requires num_sensors to be divisible by 3, got {num_channels}"
            )

        temp = torch.zeros(input_shape)
        temp_split = extract_blocks_with_overlap_torch(temp, segment_length, overlap_ratio)
        self.num_segments = temp_split.shape[-1] // num_channels
        self.temporal_flag = temporal_flag
        self.fft_flag = fft_flag
        self.share_flag = share_flag
        self.fuse_early = fuse_early
        self.overlap_ratio = overlap_ratio
        self.segment_length = segment_length
        self.feature_dim = feature_dim
        self.num_channels = num_channels
        self.grouped_channels = num_channels // 3 if fuse_early else num_channels

        if fuse_early:
            temporal_input_dim = segment_length * 3
            fft_input_dim = segment_length * 6
        else:
            temporal_input_dim = segment_length
            fft_input_dim = segment_length * 2

        self.temporal_input_dim = temporal_input_dim
        self.fft_input_dim = fft_input_dim

        if temporal_flag:
            if share_flag:
                self.temporal_weight = nn.Parameter(torch.randn(temporal_input_dim, feature_dim))
                self.temporal_bias = nn.Parameter(torch.randn(feature_dim))
            else:
                self.temporal_weight = nn.Parameter(
                    torch.randn(self.grouped_channels, temporal_input_dim, feature_dim)
                )
                self.temporal_bias = nn.Parameter(torch.randn(self.grouped_channels, feature_dim))
        else:
            self.temporal_weight = None
            self.temporal_bias = None

        if fft_flag:
            if share_flag:
                self.fft_weight = nn.Parameter(torch.randn(fft_input_dim, feature_dim))
                self.fft_bias = nn.Parameter(torch.randn(feature_dim))
            else:
                self.fft_weight = nn.Parameter(
                    torch.randn(self.grouped_channels, fft_input_dim, feature_dim)
                )
                self.fft_bias = nn.Parameter(torch.randn(self.grouped_channels, feature_dim))
        else:
            self.fft_weight = None
            self.fft_bias = None

        self.activation = nn.ReLU()
        self.norm_temporal = nn.LayerNorm(feature_dim * self.num_segments)
        self.norm_fft = nn.LayerNorm(feature_dim * self.num_segments)
        fusion_input_dim = feature_dim * 2 if fft_flag and temporal_flag else feature_dim
        self.fusion_layer = nn.Linear(fusion_input_dim, 2 * feature_dim)

    def _group_channels(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if not self.fuse_early:
            return x

        x = x.reshape(batch_size, self.num_channels, self.num_segments, -1)
        x_1 = x[:, ::3, :, :]
        x_2 = x[:, 1::3, :, :]
        x_3 = x[:, 2::3, :, :]
        return torch.cat([x_1, x_2, x_3], dim=-1)

    def _project(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self._group_channels(x)
        if self.share_flag:
            projected = torch.einsum("bcnt,tf->bcnf", x, weight) + bias
        else:
            projected = torch.einsum("bcnt,ctf->bcnf", x, weight) + bias.unsqueeze(1)
        projected = projected.reshape(batch_size, self.grouped_channels, -1)
        projected = self.activation(projected)
        if x.shape[1] != self.grouped_channels:
            raise ValueError("Unexpected grouped channel count during projection")
        return projected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, features, _, num_channels = x.shape
        if features != 1 or num_channels != self.num_channels:
            raise ValueError(f"Expected input shape (B, 1, T, {self.num_channels}), got {tuple(x.shape)}")

        x_split = extract_blocks_with_overlap_torch(x, self.segment_length, self.overlap_ratio)
        x_split = x_split.permute(0, 3, 2, 1).squeeze(-1)
        x_split = x_split.reshape(batch_size, num_channels, self.num_segments, self.segment_length)

        temporal_branch = None
        if self.temporal_flag and self.temporal_weight is not None and self.temporal_bias is not None:
            temporal_branch = self._project(x_split, self.temporal_weight, self.temporal_bias)
            temporal_branch = self.norm_temporal(temporal_branch)
            temporal_branch = temporal_branch.reshape(
                batch_size, self.grouped_channels * self.num_segments, self.feature_dim
            )

        fft_branch = None
        if self.fft_flag and self.fft_weight is not None and self.fft_bias is not None:
            fft = torch.fft.fft(x_split, dim=-1)
            fft_features = torch.cat([fft.real, fft.imag], dim=-1)
            fft_branch = self._project(fft_features, self.fft_weight, self.fft_bias)
            fft_branch = self.norm_fft(fft_branch)
            fft_branch = fft_branch.reshape(
                batch_size, self.grouped_channels * self.num_segments, self.feature_dim
            )

        if temporal_branch is not None and fft_branch is not None:
            merged = torch.cat([temporal_branch, fft_branch], dim=-1)
        elif temporal_branch is not None:
            merged = temporal_branch
        elif fft_branch is not None:
            merged = fft_branch
        else:
            raise ValueError("At least one of temporal_flag or fft_flag must be True")

        merged = merged.reshape(batch_size, self.grouped_channels, self.num_segments, -1)
        merged = self.activation(self.fusion_layer(merged))
        return merged.permute(0, 3, 2, 1)


class MLP(nn.Module):
    def __init__(self, num_features: int, expansion_factor: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_features = max(1, int(round(num_features * expansion_factor)))
        self.fc1 = nn.Linear(num_features, hidden_features)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_segments: int,
        sensor_channels: int,
        expansion_factor: float,
        dropout: float,
        use_skip_connection: bool,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_segments = num_segments
        self.sensor_channels = sensor_channels
        self.use_skip_connection = use_skip_connection
        self.norm1 = nn.LayerNorm(num_features * num_segments)
        self.mlp = MLP(num_features * num_segments, expansion_factor, dropout)
        self.norm2 = nn.LayerNorm(num_features * num_segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        residual = x
        x = x.permute(0, 3, 2, 1).reshape(batch_size, self.sensor_channels, -1)
        x = self.norm1(x)
        x = self.mlp(x)
        if self.use_skip_connection:
            x = x + residual.permute(0, 3, 2, 1).reshape(batch_size, self.sensor_channels, -1)
        x = self.norm2(x)
        return x.reshape(batch_size, self.sensor_channels, self.num_segments, self.num_features).permute(
            0, 3, 2, 1
        )


class ChannelMixer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_segments: int,
        sensor_channels: int,
        expansion_factor: float,
        dropout: float,
        use_skip_connection: bool,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_segments = num_segments
        self.sensor_channels = sensor_channels
        self.use_skip_connection = use_skip_connection
        self.mlp = MLP(num_features * sensor_channels, expansion_factor, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        residual = x
        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.num_segments, -1)
        x = self.mlp(x)
        x = x.reshape(batch_size, self.num_segments, self.num_features, self.sensor_channels).permute(0, 2, 1, 3)
        if self.use_skip_connection:
            x = x + residual
        return x


class MixerLayer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_segments: int,
        sensor_channels: int,
        expansion_factor: float,
        dropout: float,
        use_skip_connection: bool,
    ) -> None:
        super().__init__()
        self.token_mixer = TokenMixer(
            num_features=num_features,
            num_segments=num_segments,
            sensor_channels=sensor_channels,
            expansion_factor=expansion_factor,
            dropout=dropout,
            use_skip_connection=use_skip_connection,
        )
        self.channel_mixer = ChannelMixer(
            num_features=num_features,
            num_segments=num_segments,
            sensor_channels=sensor_channels,
            expansion_factor=expansion_factor,
            dropout=dropout,
            use_skip_connection=use_skip_connection,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x


class FFTMIXER_HAR_Model(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int, int],
        number_class: int,
        filter_num: int,
        fft_mixer_segments_length: int,
        expansion_factor: float,
        fft_mixer_layer_nr: int,
        fuse_early: bool,
        temporal_merge: bool,
        overlap_ratio: float,
        model_config: dict,
    ) -> None:
        super().__init__()

        _, _, window_length, sensor_channel_nr = input_shape
        self.sensor_channel_nr = sensor_channel_nr // 3 if fuse_early else sensor_channel_nr

        fft_mixer_share_flag = bool(model_config["fft_mixer_share_flag"])
        fft_mixer_temporal_flag = bool(model_config["fft_mixer_temporal_flag"])
        fft_mixer_fft_flag = bool(model_config["fft_mixer_FFT_flag"])
        use_skip_connection = bool(model_config["use_skip_connection"])

        temp = torch.zeros(input_shape)
        temp_split = extract_blocks_with_overlap_torch(temp, fft_mixer_segments_length, overlap_ratio)
        self.num_segments = temp_split.shape[-1] // sensor_channel_nr

        self.number_class = number_class
        self.filter_num = filter_num
        self.fft_mixer_segments_length = fft_mixer_segments_length
        self.fft_mixer_share_flag = fft_mixer_share_flag
        self.fft_mixer_temporal_flag = fft_mixer_temporal_flag
        self.fft_mixer_fft_flag = fft_mixer_fft_flag
        self.fft_mixer_layer_nr = fft_mixer_layer_nr
        self.expansion_factor = expansion_factor
        self.temporal_merge = temporal_merge

        self.fft_embedding_block = EmbeddingBlock(
            input_shape=input_shape,
            segment_length=fft_mixer_segments_length,
            temporal_flag=fft_mixer_temporal_flag,
            fft_flag=fft_mixer_fft_flag,
            share_flag=fft_mixer_share_flag,
            feature_dim=filter_num,
            fuse_early=fuse_early,
            overlap_ratio=overlap_ratio,
        )

        embedding_dim = filter_num * 2
        self.mixer_layer = nn.Sequential(
            *[
                MixerLayer(
                    num_features=embedding_dim,
                    num_segments=self.num_segments,
                    sensor_channels=self.sensor_channel_nr,
                    expansion_factor=expansion_factor,
                    dropout=0.0,
                    use_skip_connection=use_skip_connection,
                )
                for _ in range(fft_mixer_layer_nr)
            ]
        )

        if temporal_merge:
            self.merge = nn.Linear(embedding_dim * self.num_segments, embedding_dim)
            self.norm = nn.LayerNorm(embedding_dim)
            self.predict = nn.Linear(embedding_dim * self.sensor_channel_nr, number_class)
        else:
            self.merge = nn.Linear(embedding_dim * self.sensor_channel_nr, embedding_dim)
            self.norm = nn.LayerNorm(embedding_dim)
            self.predict = nn.Linear(embedding_dim * self.num_segments, number_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.fft_embedding_block(x)
        x = self.mixer_layer(x)

        if self.temporal_merge:
            x = x.permute(0, 3, 2, 1).reshape(batch_size, self.sensor_channel_nr, -1)
            x = F.relu(self.norm(self.merge(x)))
            x = x.reshape(batch_size, -1)
            return self.predict(x)

        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.num_segments, -1)
        x = F.relu(self.norm(self.merge(x)))
        x = x.reshape(batch_size, -1)
        return self.predict(x)


DEFAULT_CONFIG = {
    "filter_num": 6,
    "fft_mixer_segments_length": 16,
    "expansion_factor": 1.0,
    "fft_mixer_layer_nr": 2,
    "fuse_early": False,
    "temporal_merge": True,
    "overlap_ratio": 0.5,
    "fft_mixer_share_flag": False,
    "fft_mixer_temporal_flag": True,
    "fft_mixer_FFT_flag": True,
    "use_skip_connection": True,
}


class MLPHAR_Wrapper(ModelWrapper):
    NAME = "MLP-HAR"
    display_name = "MLP-HAR"
    color = "#cc4778"
    ARCHITECTURE = "FFT/time embedding + FC temporal/channel mixer"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(dense=True, spectral=True, mixer=True)
    INPUT_TYPE = "TS"
    SOURCE = "Yexu"
    NOTES = (
        "MLP-HAR implementation with FFT/time embeddings, input validation, "
        "and configurable mixer blocks."
    )
    INPUT_REQUIREMENTS = (
        "Expects 3D input shaped (B, L, C). ts_len must be at least fft_mixer_segments_length. "
        "If fuse_early=True, num_sensors must be divisible by 3."
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

        segment_length = int(model_config["fft_mixer_segments_length"])
        if ts_len < segment_length:
            raise ValueError(
                f"ts_len={ts_len} must be at least fft_mixer_segments_length={segment_length}"
            )

        self.model = FFTMIXER_HAR_Model(
            input_shape=(1, 1, ts_len, num_sensors),
            number_class=num_classes,
            filter_num=int(model_config["filter_num"]),
            fft_mixer_segments_length=segment_length,
            expansion_factor=float(model_config["expansion_factor"]),
            fft_mixer_layer_nr=int(model_config["fft_mixer_layer_nr"]),
            fuse_early=bool(model_config["fuse_early"]),
            temporal_merge=bool(model_config["temporal_merge"]),
            overlap_ratio=float(model_config["overlap_ratio"]),
            model_config=model_config,
        )

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] != self.num_sensors:
                raise ValueError(f"3D input must match (B, L, {self.num_sensors}); got {tuple(x.shape)}")
            return x.unsqueeze(1)
        if x.ndim == 4:
            if x.shape[1] != 1 or x.shape[3] != self.num_sensors:
                raise ValueError(f"4D input must match (B, 1, L, {self.num_sensors}); got {tuple(x.shape)}")
            return x
        raise ValueError(f"Expected 3D or 4D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        return self.model(x)


from whar_models._shared.adapter import ChannelFirstAdapter


def build_mlp_har(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        MLPHAR_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
