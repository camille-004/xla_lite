from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from xla_lite.core import Graph, Node, OpType, Tensor
from xla_lite.utils import validate_tensor

T = TypeVar("T")


class NodeProtocol(Protocol):
    node_id: str


class OpNode(NodeProtocol, ABC):
    @abstractmethod
    def build(self, graph: Graph) -> str:
        pass


@dataclass
class ConstantNode(OpNode):
    value: Tensor
    node_id: str = field(init=False)

    def __post_init__(self) -> None:
        validate_tensor(self.value)
        self.node_id = f"const_{id(self)}"

    def build(self, graph: Graph) -> str:
        node = Node(self.node_id, tensor=self.value, op=OpType.CONST.value)
        graph.add_node(node)
        return self.node_id


@dataclass
class BinOpNode(OpNode, Generic[T]):
    op: OpType
    left: T
    right: T
    node_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.node_id = f"{self.op.value}_{id(self)}"

    def build(self, graph: Graph) -> str:
        node = Node(
            self.node_id, op=self.op.value, inputs=[self.left, self.right]
        )
        graph.add_node(node)
        return self.node_id


class GraphBuilder:
    def __init__(self) -> None:
        self.ops: list[OpNode] = []

    def constant(self, value: Any) -> ConstantNode:
        node = ConstantNode(Tensor(value))
        self.ops.append(node)
        return node

    def add(self, a: NodeProtocol, b: NodeProtocol) -> BinOpNode:
        return self._binary_op(OpType.ADD, a, b)

    def subtract(self, a: NodeProtocol, b: NodeProtocol) -> BinOpNode:
        return self._binary_op(OpType.SUBTRACT, a, b)

    def multiply(self, a: NodeProtocol, b: NodeProtocol) -> BinOpNode:
        return self._binary_op(OpType.MULTIPLY, a, b)

    def divide(self, a: NodeProtocol, b: NodeProtocol) -> BinOpNode:
        return self._binary_op(OpType.DIVIDE, a, b)

    def matmul(self, a: NodeProtocol, b: NodeProtocol) -> BinOpNode:
        return self._binary_op(OpType.MATMUL, a, b)

    def _binary_op(
        self, op: OpType, a: NodeProtocol, b: NodeProtocol
    ) -> BinOpNode:
        node = BinOpNode(op, a.node_id, b.node_id)
        self.ops.append(node)
        return node

    def build(self) -> Graph:
        graph = Graph()
        for op in self.ops:
            op.build(graph)
        return graph
