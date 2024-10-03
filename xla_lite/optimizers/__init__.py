from .base import Optimizer, OptStrategy
from .common_subexpression_elimination import CommonSubexpressionElimination
from .constant_folding import ConstantFolding
from .dead_code_elimination import DeadCodeElimination

__all__ = [
    "Optimizer",
    "ConstantFolding",
    "CommonSubexpressionElimination",
    "DeadCodeElimination",
    "OptStrategy",
]
