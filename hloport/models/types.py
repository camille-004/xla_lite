from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable

import jax.numpy as jnp


class ActivationType(Enum):
    RELU = auto()
    TANH = auto()
    SIGMOID = auto()


@runtime_checkable
class Activation(Protocol):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray: ...


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    activation: ActivationType = ActivationType.RELU
