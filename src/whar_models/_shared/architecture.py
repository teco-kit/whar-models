from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureSpec:
    attention: bool = False
    cnn: bool = False
    dense: bool = False
    recurrent: bool = False
    transformer: bool = False
    graph: bool = False
    residual: bool = False
    spectral: bool = False
    mixer: bool = False
    feature_engineering: bool = False
    classical_ml: bool = False

    def enabled_components(self) -> tuple[str, ...]:
        return tuple(name for name, enabled in self.__dict__.items() if enabled)
