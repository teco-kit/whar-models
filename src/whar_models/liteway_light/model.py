from __future__ import annotations

from whar_models._shared.adapter import ChannelFirstAdapter
from whar_models._shared.architecture import ArchitectureSpec
from whar_models.liteway.model import _LITEWAYWrapper
from whar_models.liteway_light.liteway_light_model import LITEWAYLight as LITEWAYLightModel


class LITEWAYLight(_LITEWAYWrapper):
    NAME = "LITEWAYLight"
    display_name = "LITEWAY Light"
    color = "#43aa8b"
    ARCHITECTURE = "Depthwise-separable CNN + residual temporal convolutions + attention pooling"
    ARCHITECTURE_COMPONENTS = ArchitectureSpec(attention=True, cnn=True, dense=True)
    INPUT_TYPE = "TS"
    SOURCE = "https://github.com/dominique-nshimyimana/liteway"
    NOTES = "Upstream LITEWAY Light implementation; wrapper adds input shape conversion and class count configuration."
    INPUT_REQUIREMENTS = "Input is (B, L, sensors); L must be at least 4."
    MODEL_CLASS = LITEWAYLightModel
    MODEL_CONFIG = {"nb_conv_blocks": 4, "nb_filters": 4}


def build_liteway_light(
    *,
    input_channels: int,
    window_length: int,
    num_classes: int,
    **kwargs: object,
):
    return ChannelFirstAdapter(
        LITEWAYLight,
        input_channels=input_channels,
        window_length=window_length,
        num_classes=num_classes,
        **kwargs,
    )
