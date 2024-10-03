from __future__ import annotations

from enum import Enum
from typing import Any

from .tensor import Tensor


class OpType(Enum):
    CONST = "const"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    MATMUL = "matmul"


class Node:
    def __init__(
        self,
        node_id: Any,
        tensor: Tensor | None = None,
        op: str | None = None,
        inputs: list[Any] | None = None,
    ) -> None:
        self.node_id = node_id
        self.tensor = tensor
        self.op = op
        self.inputs = inputs or []

    def __repr__(self) -> str:
        if self.op:
            return f"Node(id='{self.node_id}', op='{self.op}', inputs={self.inputs})"  # noqa
        else:
            return f"Node(id='{self.node_id}', tensor={self.tensor})"


class Graph:
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.node_map: dict[Any, Node] = {}

    def add_node(self, node: Node) -> None:
        if node.node_id in self.node_map:
            raise ValueError(f"Node with id '{node.node_id}' already exists.")
        self.nodes.append(node)
        self.node_map[node.node_id] = node

    def get_node(self, node_id: Any) -> Node | None:
        return self.node_map.get(node_id, None)

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"{node.node_id}: {node.op}"
                if node.op
                else f"{node.node_id}: {node.tensor}"
                for node in self.nodes
            ]
        )

    def topological_sort(self) -> list[Node]:
        visited = set()
        stack = []
        temp_marks = set()

        def visit(node: Node) -> None:
            if node.node_id in temp_marks:
                raise ValueError("Graph has cycles.")
            if node.node_id not in visited:
                temp_marks.add(node.node_id)
                for input_id in node.inputs:
                    input_node = self.get_node(input_id)
                    if input_node:
                        visit(input_node)
                temp_marks.remove(node.node_id)
                visited.add(node.node_id)
                stack.append(node)

        for node in self.nodes:
            if node.node_id not in visited:
                visit(node)

        return stack
