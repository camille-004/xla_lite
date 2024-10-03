import pytest

from xla_lite.core.graph import Graph, Node, OpType, Tensor


def test_add_node_success() -> None:
    graph = Graph()
    tensor_a = Tensor([[1, 2, 3]])
    node_a = Node(node_id="A", op=OpType.CONST.value, tensor=tensor_a)
    graph.add_node(node_a)
    assert node_a in graph.nodes
    assert node_a.node_id == "A"
    assert node_a.op == OpType.CONST.value
    assert node_a.tensor.data == [[1, 2, 3]]


def test_add_duplicate_node_raises_error() -> None:
    graph = Graph()
    tensor_a = Tensor(5)
    node_a = Node(node_id="A", op=OpType.CONST.value, tensor=tensor_a)
    graph.add_node(node_a)
    with pytest.raises(ValueError, match="Node with id 'A' already exists."):
        node_duplicate = Node(
            node_id="A", op=OpType.CONST.value, tensor=Tensor(10)
        )
        graph.add_node(node_duplicate)


def test_graph_repr() -> None:
    graph = Graph()
    tensor_a = Tensor(5)
    node_a = Node(node_id="A", op=OpType.CONST.value, tensor=tensor_a)
    graph.add_node(node_a)
    expected_repr = "A: const"
    assert repr(graph) == expected_repr


def test_graph_nodes_order() -> None:
    graph = Graph()
    tensor_a = Tensor(1)
    tensor_b = Tensor(2)
    node_a = Node(node_id="A", op=OpType.CONST.value, tensor=tensor_a)
    node_b = Node(node_id="B", op=OpType.CONST.value, tensor=tensor_b)
    node_c = Node(node_id="C", op=OpType.ADD.value, inputs=["A", "B"])
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)
    exec_order = graph.topological_sort()
    assert exec_order == [node_a, node_b, node_c]
