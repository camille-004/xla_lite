from unittest.mock import Mock, patch

import pytest

from xla_lite.core import Graph, Node, OpType, Tensor
from xla_lite.frontend import BinOpNode, ConstantNode, GraphBuilder


@pytest.fixture
def graph_builder() -> GraphBuilder:
    return GraphBuilder()


def test_constant(graph_builder: GraphBuilder) -> None:
    constant_node = graph_builder.constant(5)
    assert isinstance(constant_node, ConstantNode)
    assert constant_node.value.data == 5


def test_add(graph_builder: GraphBuilder) -> None:
    a = Mock(node_id="a")
    b = Mock(node_id="b")
    add_node = graph_builder.add(a, b)
    assert isinstance(add_node, BinOpNode)
    assert add_node.op == OpType.ADD
    assert add_node.left == "a"
    assert add_node.right == "b"


def test_subtract(graph_builder: GraphBuilder) -> None:
    a = Mock(node_id="a")
    b = Mock(node_id="b")
    sub_node = graph_builder.subtract(a, b)
    assert isinstance(sub_node, BinOpNode)
    assert sub_node.op == OpType.SUBTRACT
    assert sub_node.left == "a"
    assert sub_node.right == "b"


def test_multiply(graph_builder: GraphBuilder) -> None:
    a = Mock(node_id="a")
    b = Mock(node_id="b")
    mul_node = graph_builder.multiply(a, b)
    assert isinstance(mul_node, BinOpNode)
    assert mul_node.op == OpType.MULTIPLY
    assert mul_node.left == "a"
    assert mul_node.right == "b"


def test_divide(graph_builder: GraphBuilder) -> None:
    a = Mock(node_id="a")
    b = Mock(node_id="b")
    div_node = graph_builder.divide(a, b)
    assert isinstance(div_node, BinOpNode)
    assert div_node.op == OpType.DIVIDE
    assert div_node.left == "a"
    assert div_node.right == "b"


def test_matmul(graph_builder: GraphBuilder) -> None:
    a = Mock(node_id="a")
    b = Mock(node_id="b")
    matmul_node = graph_builder.matmul(a, b)
    assert isinstance(matmul_node, BinOpNode)
    assert matmul_node.op == OpType.MATMUL
    assert matmul_node.left == "a"
    assert matmul_node.right == "b"


def test_build(graph_builder: GraphBuilder) -> None:
    with patch("xla_lite.frontend.builder.Graph") as MockGraph:
        mock_graph = MockGraph.return_value
        a = graph_builder.constant(5)
        b = graph_builder.constant(3)
        c = graph_builder.add(a, b)  # noqa: F841

        result = graph_builder.build()

        assert result == mock_graph
        assert mock_graph.add_node.call_count == 3


def test_constant_node_build() -> None:
    graph = Mock(spec=Graph)
    node = ConstantNode(Tensor(5))
    node_id = node.build(graph)

    graph.add_node.assert_called_once()
    called_with = graph.add_node.call_args[0][0]
    assert isinstance(called_with, Node)
    assert called_with.tensor is not None
    assert called_with.tensor.data == 5
    assert called_with.op == OpType.CONST.value
    assert node_id == node.node_id


def test_binary_op_node_build() -> None:
    graph = Mock(spec=Graph)
    node = BinOpNode(OpType.ADD, "a", "b")
    node_id = node.build(graph)

    graph.add_node.assert_called_once()
    called_with = graph.add_node.call_args[0][0]
    assert isinstance(called_with, Node)
    assert called_with.op == OpType.ADD.value
    assert called_with.inputs == ["a", "b"]
    assert node_id == node.node_id


def test_invalid_constant() -> None:
    with pytest.raises(ValueError):
        ConstantNode("not a tensor")  # type: ignore


def test_complex_Graph_build(graph_builder: GraphBuilder) -> None:
    with patch("xla_lite.frontend.builder.Graph") as MockGraph:
        mock_graph = MockGraph.return_value
        a = graph_builder.constant(5)
        b = graph_builder.constant(3)
        c = graph_builder.add(a, b)
        d = graph_builder.constant(2)
        e = graph_builder.multiply(c, d)  # noqa: F841

        result = graph_builder.build()

        assert result == mock_graph
        assert mock_graph.add_node.call_count == 5
