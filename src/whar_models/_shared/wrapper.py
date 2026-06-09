from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

from whar_models._shared.architecture import ArchitectureSpec


class ModelWrapper(nn.Module, ABC):
    """
    Unified wrapper API for HAR models.

    Required constructor args:
    - num_sensors
    - num_classes

    External input must be provided as (batch, ts_len, num_sensors).
    Wrappers may internally expand that into 4D BCHW-like layout:
    (batch, 1, ts_len, num_sensors).
    """

    NAME: str | None = None
    display_name: str | None = None
    color: str | None = None
    ARCHITECTURE: str | None = None
    INPUT_TYPE: str = "TS"
    SOURCE: str = ""
    NOTES: str = ""
    INPUT_REQUIREMENTS: str = ""
    ARCHITECTURE_COMPONENTS: ArchitectureSpec = ArchitectureSpec()

    def __init__(self, num_sensors: int, num_classes: int) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.architecture = self.get_architecture_components()

    @abstractmethod
    def to_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Convert/validate external input into the shape expected by the wrapped model."""
        raise NotImplementedError

    def get_name(self) -> str:
        name = getattr(self, "NAME", None)
        if isinstance(name, str) and name.strip():
            return name
        return self.__class__.__name__

    def get_display_name(self) -> str:
        display_name = getattr(self, "display_name", None)
        if isinstance(display_name, str) and display_name.strip():
            return display_name
        name = getattr(self, "NAME", None)
        if isinstance(name, str) and name.strip():
            return name
        return self.get_name()

    def get_color(self) -> str:
        color = getattr(self, "color", None)
        if isinstance(color, str) and color.strip():
            return color
        return "#b24a2f"

    def get_architecture(self) -> str:
        arch = getattr(self, "ARCHITECTURE", None)
        if isinstance(arch, str) and arch.strip():
            return arch
        return self.get_name()

    def get_architecture_components(self) -> ArchitectureSpec:
        components = getattr(self, "ARCHITECTURE_COMPONENTS", ArchitectureSpec())
        if isinstance(components, ArchitectureSpec):
            return components
        raise TypeError(
            f"{self.__class__.__name__}.ARCHITECTURE_COMPONENTS must be an ArchitectureSpec instance."
        )

    def get_input_type(self) -> str:
        input_type = getattr(self, "INPUT_TYPE", "TS")
        if isinstance(input_type, str) and input_type.strip():
            return input_type
        return "TS"

    def get_source(self) -> str:
        source = getattr(self, "SOURCE", "")
        if isinstance(source, str):
            return source
        return ""

    def get_notes(self) -> str:
        notes = getattr(self, "NOTES", "")
        if isinstance(notes, str):
            return notes
        return ""

    def get_input_requirements(self) -> str:
        requirements = getattr(self, "INPUT_REQUIREMENTS", "")
        if isinstance(requirements, str):
            return requirements
        return ""

    def get_trainable_param_count(self) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter)
        )
