from __future__ import annotations

import torch
import torch.nn as nn

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=(1, 0),
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self, temperature):
        super(AttentionGate, self).__init__()
        kernel_size = (5, 1)
        self.temperature = temperature
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(2, 0), relu=False)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print("Change temperature to:", str(self.temperature))

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False, temperature=34):
        super(TripletAttention, self).__init__()

        self.cw = AttentionGate(temperature)
        self.hc = AttentionGate(temperature)
        self.no_spatial = no_spatial

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.w1.data.fill_(1 / 3)
        self.w2.data.fill_(1 / 3)
        self.w3.data.fill_(1 / 3)

        if not no_spatial:
            self.hw = AttentionGate(temperature)

    def update_temperature(self):
        self.cw.updata_temperature()
        self.hc.updata_temperature()
        self.hw.updata_temperature()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = self.w1 * x_out + self.w2 * x_out11 + self.w3 * x_out21
        else:
            x_out = self.w1 * x_out11 + self.w2 * x_out21
        return x_out


class TripleAttentionCNN(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(TripleAttentionCNN, self).__init__()
        _ = input_shape
        conv1 = nn.Conv2d(1, 64, (6, 1), (3, 1), padding=(1, 0))
        att1 = TripletAttention()

        conv2 = nn.Conv2d(64, 128, (6, 1), (3, 1), padding=(1, 0))
        att2 = TripletAttention()

        conv3 = nn.Conv2d(128, 256, (6, 1), (3, 1), padding=(1, 0))
        att3 = TripletAttention()

        self.conv_module = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            att1,
            conv2,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            att2,
            conv3,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            att3,
        )
        flattened_dim = self._infer_flattened_dim(input_shape)
        self.classifier = nn.Sequential(nn.Linear(flattened_dim, nb_classes))

    def _infer_flattened_dim(self, input_shape) -> int:
        with torch.no_grad():
            dummy = torch.zeros(*input_shape)
            out = self.conv_module(dummy)
            return int(out.flatten(1).shape[1])

    def forward(self, x):
        x = self.conv_module(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TripleAttentionResNet(nn.Module):
    def __init__(self, input_shape, nb_classes):
        super(TripleAttentionResNet, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.att1 = TripletAttention()

        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
        )
        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.att2 = TripletAttention()

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )
        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.att3 = TripletAttention()

        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
        )
        flattened_dim = self._infer_flattened_dim(input_shape)
        self.fc = nn.Sequential(nn.Linear(flattened_dim, nb_classes))
        self.out_norm = nn.LayerNorm(nb_classes)

    def _infer_flattened_dim(self, input_shape) -> int:
        with torch.no_grad():
            dummy = torch.zeros(*input_shape)
            out1 = self.Block1(dummy)
            out1 = self.att1(out1)
            y1 = self.shortcut1(dummy)
            out = y1 + out1

            out2 = self.Block2(out)
            out2 = self.att2(out2)
            y2 = self.shortcut2(out)
            out = y2 + out2

            out3 = self.Block3(out)
            out3 = self.att3(out3)
            y3 = self.shortcut3(out)
            out = y3 + out3
            return int(out.flatten(1).shape[1])

    def forward(self, x):
        out1 = self.Block1(x)
        out1 = self.att1(out1)
        y1 = self.shortcut1(x)
        out = y1 + out1

        out2 = self.Block2(out)
        out2 = self.att2(out2)
        y2 = self.shortcut2(out)
        out = y2 + out2

        out3 = self.Block3(out)
        out3 = self.att3(out3)
        y3 = self.shortcut3(out)
        out = y3 + out3

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.out_norm(out)
        return out

class TripleCrossDomainAttention_Wrapper(ModelWrapper):
    NAME = "TripleCrossDomainAttention"
    display_name = "Triple-Cross-Attn"
    color = "#ed7953"
    ARCHITECTURE = "3x Conv2D + triplet cross-domain attention"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(attention=True, cnn=True, dense=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/yinntag/Triple-Cross-domain-Attention-for-HAR/tree/main/Model"
    NOTES = (
        "Implements both backbones defined in Model/main.py using the original "
        "TripletAttention block from Model/trip_att.py. Select the backbone with "
        "the wrapper `backbone` flag: `cnn` or `resnet`."
    )

    def __init__(
        self,
        num_sensors: int,
        num_classes: int,
        ts_len: int = 128,
        backbone: str = "cnn",
        **kwargs,
    ) -> None:
        super().__init__(num_sensors=num_sensors, num_classes=num_classes)
        _ = kwargs
        self.backbone = backbone.lower()
        input_shape = (1, 1, ts_len, num_sensors)

        if self.backbone == "cnn":
            self.model = TripleAttentionCNN(
                input_shape=input_shape,
                nb_classes=num_classes,
            )
        elif self.backbone == "resnet":
            self.model = TripleAttentionResNet(
                input_shape=input_shape,
                nb_classes=num_classes,
            )
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Expected one of: 'cnn', 'resnet'."
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


def build_triple_cross_domain_attention(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        TripleCrossDomainAttention_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
