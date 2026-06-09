from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from whar_models._shared.architecture import ArchitectureSpec
from whar_models._shared.wrapper import ModelWrapper


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {class_label: np.identity(len(classes))[i, :] for i, class_label in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def edge_init(node_num: int) -> tuple[torch.Tensor, torch.Tensor]:
    off_diag = np.ones([node_num, node_num]) - np.eye(node_num)
    rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

    relation_num = node_num - 1
    rel_rec_undirected = np.empty([0, node_num])
    rel_send_undirected = np.empty([0, node_num])
    for k in range(1, relation_num + 1):
        rel_rec_undirected = np.concatenate(
            (rel_rec_undirected, rel_rec[((k - 1) * relation_num + k - 1) : (k * relation_num), :]),
            axis=0,
        )
        rel_send_undirected = np.concatenate(
            (rel_send_undirected, rel_send[((k - 1) * relation_num + k - 1) : (k * relation_num), :]),
            axis=0,
        )

    return torch.FloatTensor(rel_rec_undirected), torch.FloatTensor(rel_send_undirected)


class _DynamicWHAR(nn.Module):
    def __init__(
        self,
        node_num=5,
        node_dim=9,
        window_size=24,
        channel_dim=8,
        time_reduce_size=10,
        hid_dim=128,
        class_num=17,
    ):
        super().__init__()
        self.node_num = node_num
        self.node_dim = node_dim
        self.window_size = window_size
        self.channel_dim = channel_dim
        self.time_reduce_size = time_reduce_size
        self.hid_dim = hid_dim
        self.class_num = class_num

        self.dropout_prob = 0.6
        self.conv1 = nn.Conv1d(self.node_dim, self.channel_dim, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(self.channel_dim)
        self.conv2 = nn.Conv1d(self.window_size, self.time_reduce_size, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(self.time_reduce_size)
        self.conv3 = nn.Conv1d(
            self.channel_dim * self.time_reduce_size * 2,
            self.channel_dim * self.time_reduce_size * 2,
            kernel_size=1,
            stride=1,
        )
        self.bn3 = nn.BatchNorm1d(self.channel_dim * self.time_reduce_size * 2)
        self.conv5 = nn.Conv1d(
            self.channel_dim * self.time_reduce_size * 2,
            self.channel_dim * self.time_reduce_size * 2,
            kernel_size=1,
            stride=1,
        )
        self.bn5 = nn.BatchNorm1d(self.channel_dim * self.time_reduce_size * 2)

        self.msg_fc1 = nn.Linear(self.channel_dim * self.time_reduce_size * 3 * self.node_num, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.class_num)

        self.conv4 = nn.Conv1d(self.channel_dim * self.time_reduce_size * 2, 1, kernel_size=1, stride=1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def edge2node(self, x, rel_rec, rel_send, rel_type):
        mask = rel_type.squeeze(-1)
        x = x + x * (mask.unsqueeze(2))
        rel = rel_rec.t() + rel_send.t()
        incoming = torch.matmul(rel, x)
        return incoming / incoming.size(1)

    def forward(self, inputs, rel_rec, rel_send):
        x = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(inputs.shape[0], inputs.shape[1], x.shape[1])
        s_input_1 = x

        edge = self.node2edge(s_input_1, rel_rec, rel_send)
        edge = edge.permute(0, 2, 1)
        edge = F.relu(self.bn3(self.conv3(edge)))
        edge = edge.permute(0, 2, 1)

        x = edge.permute(0, 2, 1)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        rel_type = F.sigmoid(x)

        s_input_2 = self.edge2node(edge, rel_rec, rel_send, rel_type)
        s_input_2 = s_input_2.permute(0, 2, 1)
        s_input_2 = F.relu(self.bn5(self.conv5(s_input_2)))
        s_input_2 = s_input_2.permute(0, 2, 1)

        join = torch.cat((s_input_1, s_input_2), dim=2)
        join = join.reshape(join.shape[0], -1)
        join = F.dropout(join, p=self.dropout_prob, training=self.training)
        join = F.relu(self.msg_fc1(join))
        join = F.dropout(join, p=self.dropout_prob, training=self.training)
        preds = self.fc_out(join)

        return preds


class DynamicWHAR_Wrapper(ModelWrapper):
    NAME = "DynamicWHAR"
    display_name = "DynamicWHAR"
    color = "#b5179e"
    ARCHITECTURE = "Per-sensor CNN + dynamic graph interaction"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(cnn=True, dense=True, graph=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/wdkhuans/DynamicWHAR"
    NOTES = "DynamicWHAR implementation with per-sensor temporal encoding and graph interaction."
    INPUT_REQUIREMENTS = (
        "Input is reshaped to (B, node_num, ts_len, node_dim). "
        "For generic benchmark tensors, the wrapper defaults to node_dim=1 unless configured explicitly."
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
        model_config = {} if config is None else dict(config)
        model_config.update(kwargs)

        node_dim = int(model_config.get("node_dim", self._infer_node_dim(num_sensors)))
        if num_sensors % node_dim != 0:
            raise ValueError(f"num_sensors={num_sensors} must be divisible by node_dim={node_dim}")
        node_num = int(model_config.get("node_num", num_sensors // node_dim))

        if node_num * node_dim != num_sensors:
            raise ValueError(
                f"node_num * node_dim must equal num_sensors; got {node_num} * {node_dim} != {num_sensors}"
            )

        self.node_num = node_num
        self.node_dim = node_dim
        self.window_size = ts_len
        self.model = _DynamicWHAR(
            node_num=node_num,
            node_dim=node_dim,
            window_size=ts_len,
            channel_dim=int(model_config.get("channel_dim", 32)),
            time_reduce_size=int(model_config.get("time_reduce_size", 8)),
            hid_dim=int(model_config.get("hid_dim", 128)),
            class_num=num_classes,
        )
        rel_rec, rel_send = edge_init(self.node_num)
        self.register_buffer("rel_rec", rel_rec)
        self.register_buffer("rel_send", rel_send)

    def _infer_node_dim(self, num_sensors: int) -> int:
        _ = num_sensors
        return 1

    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[2] != self.num_sensors:
                raise ValueError(f"3D input must match (B, L, {self.num_sensors}); got {tuple(x.shape)}")
            return x.reshape(x.shape[0], x.shape[1], self.node_num, self.node_dim).permute(0, 2, 1, 3)
        if x.ndim == 4:
            expected = (self.node_num, self.window_size, self.node_dim)
            if tuple(x.shape[1:]) != expected:
                raise ValueError(f"4D input must match (B, {expected[0]}, {expected[1]}, {expected[2]}); got {tuple(x.shape)}")
            return x
        raise ValueError(f"Expected 3D or 4D input tensor, got {x.ndim}D input with shape {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_input_shape(x)
        return self.model(x, self.rel_rec, self.rel_send)


from whar_models._shared.adapter import ChannelFirstAdapter


def build_dynamic_whar(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        DynamicWHAR_Wrapper,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
