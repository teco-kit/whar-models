from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


DEFAULT_CONFIG = {
    "dff": 512,
    "d_model": 128,
    "nh": 4,
    "dropout_rate": 0.2,
    "use_pe": True,
    "sensor_n_filters": 128,
    "sensor_kernel_size": 3,
    "sensor_dilation_rate": 2,
}


class AttentionWithContext(nn.Module):
    def __init__(self, bias: bool = True, return_attention: bool = False):
        super().__init__()
        self.bias = bias
        self.return_attention = return_attention
        self.W: nn.Parameter | None = None
        self.b: nn.Parameter | None = None
        self.u: nn.Parameter | None = None

    def _build(self, input_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        self.W = nn.Parameter(torch.empty(input_dim, input_dim, device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.W)
        if self.bias:
            self.b = nn.Parameter(torch.zeros(input_dim, device=device, dtype=dtype))
        self.u = nn.Parameter(torch.empty(input_dim, device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.u.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        if self.W is None or self.u is None:
            self._build(input_dim=x.shape[-1], device=x.device, dtype=x.dtype)

        uit = torch.tensordot(x, self.W, dims=([2], [0]))
        if self.bias and self.b is not None:
            uit = uit + self.b
        uit = torch.tanh(uit)
        ait = torch.tensordot(uit, self.u, dims=([2], [0]))

        a = torch.exp(ait)
        a = a / (a.sum(dim=1, keepdim=True) + torch.finfo(x.dtype).eps)
        a = a.unsqueeze(-1)
        weighted_input = x * a
        result = weighted_input.sum(dim=1)

        if self.return_attention:
            return result, a
        return result


class SensorAttention(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_filters: int,
        kernel_size: int,
        dilation_rate: int,
    ):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            1,
            n_filters,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding="same",
        )
        self.conv_f = nn.Conv2d(n_filters, 1, kernel_size=1, padding="same")
        self.ln = nn.LayerNorm(normalized_shape=n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.ln(x)
        x1 = x.unsqueeze(1)
        x1 = F.relu(self.conv_1(x1))
        x1 = self.conv_f(x1)
        x1 = F.softmax(x1, dim=3)
        x1 = x1.squeeze(1)
        return x * x1, x1


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = torch.tensor(k.shape[-1], device=q.device, dtype=q.dtype)
    scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=True)
        self.wv = nn.Linear(d_model, d_model, bias=True)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        v: torch.Tensor,
        k: torch.Tensor,
        q: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)
        return output, attention_weights


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, dff: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x: torch.Tensor, training: bool = False, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output if training else attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output if training else ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class PositionalEncoding(nn.Module):
    def __init__(self, position: int, d_model: int):
        super().__init__()
        self.register_buffer("pos_encoding", self.positional_encoding(position, d_model))

    def get_angles(self, position: torch.Tensor, i: torch.Tensor, d_model: int) -> torch.Tensor:
        angles = 1 / torch.pow(
            torch.tensor(10000.0, dtype=torch.float32, device=position.device),
            (2 * torch.floor(i / 2)) / float(d_model),
        )
        return position * angles

    def positional_encoding(self, position: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(position, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        angle_rads = self.get_angles(pos, i, d_model)
        sines = torch.sin(angle_rads[:, 0::2])
        cosines = torch.cos(angle_rads[:, 1::2])
        pos_encoding = torch.cat([sines, cosines], dim=-1).unsqueeze(0)
        return pos_encoding.to(torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.pos_encoding[:, : inputs.shape[1], :]


class SAHARModel(nn.Module):
    def __init__(
        self,
        n_timesteps: int,
        n_features: int,
        n_outputs: int,
        _dff: int = 512,
        d_model: int = 128,
        nh: int = 4,
        dropout_rate: float = 0.2,
        use_pe: bool = True,
        sensor_n_filters: int = 128,
        sensor_kernel_size: int = 3,
        sensor_dilation_rate: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pe = use_pe

        self.sensor_attention = SensorAttention(
            n_features=n_features,
            n_filters=sensor_n_filters,
            kernel_size=sensor_kernel_size,
            dilation_rate=sensor_dilation_rate,
        )
        self.proj = nn.Conv1d(n_features, d_model, kernel_size=1)
        self.positional_encoding = PositionalEncoding(n_timesteps, d_model)
        self.pos_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = EncoderLayer(d_model=d_model, num_heads=nh, dff=_dff, rate=dropout_rate)
        self.encoder2 = EncoderLayer(d_model=d_model, num_heads=nh, dff=_dff, rate=dropout_rate)
        self.attentive_pool = AttentionWithContext()
        self.fc1 = nn.Linear(d_model, n_outputs * 4)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(n_outputs * 4, n_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        si, _ = self.sensor_attention(inputs)

        x = self.proj(si.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)

        if self.use_pe:
            x = x * math.sqrt(float(self.d_model))
            x = self.positional_encoding(x)
            x = self.pos_dropout(x)

        x = self.encoder1(x, training=self.training)
        x = self.encoder2(x, training=self.training)
        x = self.attentive_pool(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class SA_HAR_Wrapper(ModelWrapper):
    NAME = "SA_HAR"
    display_name = "SA-HAR"
    color = "#de4968"
    ARCHITECTURE = "sensor attention + Conv1D + 2x self-attention encoder + attentive pooling"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(attention=True, cnn=True, dense=True, transformer=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/saif-mahmud/self-attention-HAR"
    NOTES = "PyTorch port that preserves the original TensorFlow model block structure."

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
        self.model = SAHARModel(
            n_timesteps=ts_len,
            n_features=num_sensors,
            n_outputs=num_classes,
            _dff=model_config["dff"],
            d_model=model_config["d_model"],
            nh=model_config["nh"],
            dropout_rate=model_config["dropout_rate"],
            use_pe=model_config["use_pe"],
            sensor_n_filters=model_config["sensor_n_filters"],
            sensor_kernel_size=model_config["sensor_kernel_size"],
            sensor_dilation_rate=model_config["sensor_dilation_rate"],
        )

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


def build_sa_har(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        SA_HAR_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
