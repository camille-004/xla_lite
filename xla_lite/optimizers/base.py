import abc
from functools import wraps
from typing import Callable

from xla_lite.core import Graph, OpType, Tensor
from xla_lite.core.ops import add, divide, matmul, multiply, subtract


def timing(func):
    @wraps
    def wrapper(*args, **kwargs):
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class OptStrategy(abc.ABC):
    @abc.abstractmethod
    def apply(self, graph: Graph) -> None:
        pass


class Optimizer:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    @staticmethod
    def _execute_operation(op: str, inputs: list[Tensor]) -> Tensor:
        op_map: dict[str, Callable] = {
            OpType.ADD.value: add,
            OpType.SUBTRACT.value: subtract,
            OpType.MULTIPLY.value: multiply,
            OpType.DIVIDE.value: divide,
            OpType.MATMUL.value: matmul,
        }
        if op in op_map:
            return op_map[op](*inputs)
        else:
            raise ValueError(f"Unsupported operation for optimization: {op}")
