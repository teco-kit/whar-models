import torch.nn as nn
import torch
from torch.nn import functional as F

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper

"""
Implementation of the Attend-and-Discriminate model, from here:
https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate
"""

DEFAULT_DROPOUT = 0.5
DEFAULT_DROPOUT_RNN = 0.25
DEFAULT_DROPOUT_CLS = 0.5

# utils.utils_attention.py
def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    """
    Create and initialize a `nn.Conv1d` layer with spectral normalization.
    """
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    # return spectral_norm(conv)
    return conv


class SelfAttention(nn.Module):
    """
    # self-attention implementation from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    Self attention layer for nd
    """
    def __init__(self, n_channels: int, div):
        super(SelfAttention, self).__init__()

        if n_channels > 1:
            self.query = conv1d(n_channels, n_channels//div)
            self.key = conv1d(n_channels, n_channels//div)
        else:
            self.query = conv1d(n_channels, n_channels)
            self.key = conv1d(n_channels, n_channels)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class TemporalAttention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        out = self.fc(x).squeeze(2)
        weights_att = self.sm(out).unsqueeze(2)
        context = torch.sum(weights_att * x, 0)
        return context


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        activation,
        sa_div,
    ):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, filter_num, (filter_size, 1))
        self.conv2 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv3 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.conv4 = nn.Conv2d(filter_num, filter_num, (filter_size, 1))
        self.activation = nn.ReLU() if activation == "ReLU" else nn.Tanh()

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            filter_num * input_dim,
            hidden_dim,
            enc_num_layers,
            bidirectional=enc_is_bidirectional,
            dropout=dropout_rnn,
        )

        self.ta = TemporalAttention(hidden_dim)
        self.sa = SelfAttention(filter_num, sa_div)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # apply self-attention on each temporal dimension (along sensor and feature dimensions)
        refined = torch.cat(
            [self.sa(torch.unsqueeze(x[:, :, t, :], dim=3)) for t in range(x.shape[2])],
            dim=-1,
        )
        x = refined.permute(3, 0, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        outputs, h = self.rnn(x)

        # apply temporal attention on GRU outputs
        out = self.ta(outputs)
        return out


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_class):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_class)

    def forward(self, z):
        return self.fc(z)


class AttendDiscriminateModel(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        input_dim,
        hidden_dim,
        filter_num,
        filter_size,
        enc_num_layers,
        enc_is_bidirectional,
        dropout,
        dropout_rnn,
        dropout_cls,
        activation,
        sa_div,
        num_class,
        train_mode,
        experiment,
    ):
        super(AttendDiscriminateModel, self).__init__()

        self.experiment = f"train_{experiment}" if train_mode else experiment
        self.model = model
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        print(f"[STEP 3] Creating {self.model} HAR model ...")

        self.fe = FeatureExtractor(
            input_dim,
            hidden_dim,
            filter_num,
            filter_size,
            enc_num_layers,
            enc_is_bidirectional,
            dropout,
            dropout_rnn,
            activation,
            sa_div,
        )

        self.dropout = nn.Dropout(dropout_cls)
        self.classifier = Classifier(hidden_dim, num_class)
        # Keep centers device-agnostic; module/device movement handles placement.
        self.register_buffer("centers", torch.randn(num_class, self.hidden_dim))

    def forward(self, x):
        feature = self.fe(x)
        z = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature)
        )
        out = self.dropout(feature)
        logits = self.classifier(out)
        return z, logits

    @property
    def path_checkpoints(self):
        return f"./models/{self.dataset}/{self.experiment}/checkpoints/"

    @property
    def path_logs(self):
        return f"./models/{self.dataset}/{self.experiment}/logs/"

    @property
    def path_visuals(self):
        return f"./models/{self.dataset}/{self.experiment}/visuals/"


class AttendDiscriminate(ModelWrapper):
    NAME = "AttendDiscriminate"
    display_name = "Attend+Discriminate"
    color = "#ff7f11"
    ARCHITECTURE = "Conv2D + GRU + attention"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(attention=True, cnn=True, dense=True, recurrent=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate"
    NOTES = "Uses one benchmark-wide dropout configuration instead of dataset-specific tuning."

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        ts_len: int = 128,
        hidden_dim: int = 128,
        filter_num: int = 64,
        filter_size: int = 5,
        enc_num_layers: int = 2,
        enc_is_bidirectional: bool = False,
        dropout: float | None = None,
        dropout_rnn: float | None = None,
        dropout_cls: float | None = None,
        activation: str = "ReLU",
        sa_div: int = 1,
        model: str = "AttendDiscriminate",
        dataset: str = "default",
        train_mode: bool = False,
        experiment: str = "default",
        **kwargs,
    ) -> None:
        super().__init__(num_sensors=num_sensors, num_classes=num_classes)
        self.ts_len = ts_len

        dropout = DEFAULT_DROPOUT if dropout is None else dropout
        dropout_rnn = DEFAULT_DROPOUT_RNN if dropout_rnn is None else dropout_rnn
        dropout_cls = DEFAULT_DROPOUT_CLS if dropout_cls is None else dropout_cls

        self.model = AttendDiscriminateModel(
            model=model,
            dataset=dataset,
            input_dim=num_sensors,
            hidden_dim=hidden_dim,
            filter_num=filter_num,
            filter_size=filter_size,
            enc_num_layers=enc_num_layers,
            enc_is_bidirectional=enc_is_bidirectional,
            dropout=dropout,
            dropout_rnn=dropout_rnn,
            dropout_cls=dropout_cls,
            activation=activation,
            sa_div=sa_div,
            num_class=num_classes,
            train_mode=train_mode,
            experiment=experiment,
            **kwargs,
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
        # Underlying architecture expects (B, L, C), wrapper API standardizes to (B, 1, L, C).
        x = x.squeeze(1)
        _, logits = self.model(x)
        return logits


from whar_models._shared.adapter import ChannelFirstAdapter


def build_attend_discriminate(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        AttendDiscriminate,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
