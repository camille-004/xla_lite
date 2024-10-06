import struct
from typing import Any

from xla_lite.core import Data, Graph, Node, OpType
from xla_lite.optimizers import OptStrategy


class CommonSubexpressionElimination(OptStrategy):
    def apply(self, graph: Graph) -> None:
        op_signature_map: dict[tuple[str, tuple[Any, ...]], str] = {}

        for node in graph.nodes:
            if node.op:
                signature = self._get_node_signature(node)

                if signature in op_signature_map:
                    original_node_id = op_signature_map[signature]

                    print(
                        f"Common subexpression detected: '{node.node_id}' is "
                        + f"duplicate of '{original_node_id}'"
                    )

                    node.inputs = [original_node_id]
                    node.op = OpType.CONST.value
                    node.tensor = None

                    for other_node in graph.nodes:
                        other_node.inputs = [
                            original_node_id
                            if input_id == node.node_id
                            else input_id
                            for input_id in other_node.inputs
                        ]

                    print(
                        f"Node '{node.node_id}' replaced with constant node "
                        + "'{original_node_id}'"
                    )
                else:
                    op_signature_map[signature] = node.node_id
                    print(
                        f"Registering node '{node.node_id}' with signature "
                        + "{signature}"
                    )

    @staticmethod
    def _get_node_signature(node: Node) -> tuple:
        if node.op == OpType.CONST.value and node.tensor:
            return (
                node.op,
                CommonSubexpressionElimination._data_to_bytes(
                    node.tensor.data
                ),
            )
        elif node.op in {OpType.ADD.value, OpType.MULTIPLY.value}:
            return (node.op, tuple(sorted(node.inputs)))
        else:
            return (node.op, tuple(node.inputs))

    @staticmethod
    def _data_to_bytes(data: Data) -> bytes:
        if isinstance(data, int):
            return data.to_bytes(
                (data.bit_length() + 7) // 8, byteorder="big", signed=True
            )
        elif isinstance(data, float):
            return struct.pack("!d", data)
        elif isinstance(data, list):
            return b"".join(
                CommonSubexpressionElimination._data_to_bytes(item)
                for item in data
            )
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
