import torch
import torch.nn as nn



class ResidualStridedDownsampleStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, shortcut='', activation=''):
        super().__init__()

        padding = (dilation * (kernel_size - 1) + 1) // 2

        # Depthwise with stride=2 (downsample here)
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            (kernel_size, 1),
            stride=(2, 1),                      # 👈 downsample here
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=in_channels,
            bias=True
        )

        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)   # ReLU

        # Residual projection with stride
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=(2,1), bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # obileNet-style blocks always: spatial filtering first → downsample → channel mixing
        out = self.dw(x)
        out = self.pw(out)
        out = self.bn(out)
        out = self.act(out)
        res = self.res(x)
        return out + res



class SEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        assert in_channels == out_channels, "Make sure channels match!"

        # Main convolution path
        padding = (dilation * (kernel_size - 1) + 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, (kernel_size, 1),
            padding=(padding, 0),                                   # Maintain temporal length
            dilation=(dilation, 1),                                 # Add dilation here
            groups=in_channels                                      # depthwise
        )

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_feats = self.depthwise(x)
        # x_feats = self.se(x_feats)
        x_feats = self.bn(x_feats)
        x_feats = self.relu(x_feats)
        return x_feats



class ResidualSCTM(nn.Module):
    """Residual (Highway w/o gate) Structure Convolutional Temporal Modeling(SCTM) for sequence processing."""

    def __init__(
        self,
        in_feat,
        hidden_size=16,
        kernel_size=5,
        dilation=1,
        scale_factor=2,
    ):
        super().__init__()

        self.scale_factor = scale_factor
        padding = (kernel_size // 2) * dilation

        # Depthwise temporal conv (local temporal modeling)
        self.dw = nn.Conv1d(
            in_feat,
            in_feat,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_feat,
            bias=False,
        )

        self.norm = nn.BatchNorm1d(in_feat)
        self.act = nn.GELU()
        self.elu = nn.ELU()

        # 1. Feature mixing (like GRU candidate state) 2. GRU-like update gate 3. Residual projection
        self.pw_gate_res = nn.Conv1d(
            in_feat,
            hidden_size * self.scale_factor,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        """
        x: (B, F, T)
        """
        # x = x.transpose(1, 2)
        h = self.dw(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.pw_gate_res(h)
        out = torch.cat([h, self.elu(self.pw_gate_res(x))], dim=1)
        return out.transpose(1, 2)



class DepthwiseConv1dTemporalPooling(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Conv1d(feature_dim, 1, kernel_size=1)    #, groups=feature_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        scores = self.conv(x)
        weights = torch.softmax(scores, dim=2)
        pooled = (x * weights).sum(dim=2)
        return pooled



class LITEWAYLight(nn.Module):
    def __init__(self, input_shape, nb_classes, config={}):
        super().__init__()
        self.input_channels = input_shape[3]
        self.seq_length = input_shape[2]
        self.nb_conv_blocks = config.get('nb_conv_blocks', 4)
        self.nb_units_gru = config.get('nb_units_gru', 16)
        self.nb_filters = config.get('nb_filters', 4)
        self.drop_prob = config.get('drop_prob', 0.3)
        self.nb_classes = nb_classes

        # Scale Factor
        self.scale_factor = config.get('scale_factor', 2)    # default 2

        # ---- Stem blocks (with residual & max-pool) ----
        stem_blocks = [
            ResidualStridedDownsampleStem(1, self.nb_filters, kernel_size=5, dilation=1),
            ResidualStridedDownsampleStem(self.nb_filters, 2 * self.nb_filters, kernel_size=5, dilation=1),
        ]
        # ---- Repeated bottleneck blocks (no max-pool) ----
        bottleneck_blocks = [
            SEConvBlock(2 * self.nb_filters, 2 * self.nb_filters, kernel_size=5, dilation=1)
            for _ in range(self.nb_conv_blocks)
        ]
        # ---- Feature Extraction & Refinement: Combine all blocks ----
        self.conv_blocks =  nn.Sequential(*(stem_blocks + bottleneck_blocks))

        # ---Dropout--- #
        self.dropout = nn.Dropout(self.drop_prob)

        # ---Structured Conv. Temp. Modeling (SCTM) --- #
        self.sctm_residual = ResidualSCTM(
            in_feat=self.input_channels * 2 * self.nb_filters,
            hidden_size=self.nb_units_gru // 2,
            scale_factor=self.scale_factor,
        )

        # ---Aggregation--- #
        self.attention_pool = DepthwiseConv1dTemporalPooling(feature_dim=self.scale_factor * self.nb_units_gru)

        # ---Classifier - Linear--- #
        self.classifier = nn.Linear(self.scale_factor * self.nb_units_gru, self.nb_classes)


    def forward(self, x):
        x = self.conv_blocks(x)
        B, C, T, C_in = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, -1, T)
        x = self.dropout(x)
        x = self.sctm_residual(x)
        x = self.attention_pool(x)
        return self.classifier(x)

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)