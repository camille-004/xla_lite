from typing import Any

from ..core import Graph, OpType, Tensor
from ..core.ops import add, divide, matmul, multiply, subtract


class Executor:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.tensor_vals: dict[Any, Tensor] = {}

    def execute(self) -> dict[Any, Tensor]:
        exec_order = self.graph.topological_sort()

        for node in exec_order:
            if node.op is None or node.op == OpType.CONST.value:
                self.tensor_vals[node.node_id] = node.tensor
            else:
                try:
                    input_tensors = [
                        self.tensor_vals[input_id] for input_id in node.inputs
                    ]
                except KeyError as e:
                    raise ValueError(
                        f"Missing input tensor for node '{node.node_id}': {e}"
                    )

                result_tensor = self.exec_op(node.op, input_tensors)
                self.tensor_vals[node.node_id] = result_tensor

        return self.tensor_vals

    def exec_op(self, op: str, inputs: list) -> Tensor:
        if op == OpType.ADD.value:
            return add(*inputs)
        elif op == OpType.SUBTRACT.value:
            return subtract(*inputs)
        elif op == OpType.MULTIPLY.value:
            return multiply(*inputs)
        elif op == OpType.DIVIDE.value:
            return divide(*inputs)
        elif op == OpType.MATMUL.value:
            return matmul(*inputs)
        else:
            raise ValueError(f"Unsupported operation: {op}")
