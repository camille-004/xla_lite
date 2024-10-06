from graphviz import Digraph  # type: ignore

from xla_lite.core import Graph, Node, OpType, Tensor
from xla_lite.visualization import visualize_graph


def test_visualize_graph() -> None:
    graph = Graph()

    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    c = Node("c", op=OpType.ADD.value, inputs=["a", "b"])

    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)

    dot = visualize_graph(graph)

    assert isinstance(dot, Digraph)
    assert any(
        'a [label="a\nTensor(data=5, shape=())" shape=box]' in line
        for line in dot.body
    )
    assert any(
        'b [label="b\nTensor(data=3, shape=())" shape=box]' in line
        for line in dot.body
    )
    assert any('c [label="c\nadd" shape=ellipse]' in line for line in dot.body)
    assert any("a -> c" in line for line in dot.body)
    assert any("b -> c" in line for line in dot.body)
    assert any("rankdir=TB" in line for line in dot.body)
    assert len(dot.body) >= 6


def test_graph_visualize_method() -> None:
    graph = Graph()

    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    c = Node("c", op=OpType.ADD.value, inputs=["a", "b"])

    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)

    assert hasattr(
        graph, "visualize"
    ), "Graph class should have a visualize method"

    dot = graph.visualize()
    assert isinstance(dot, Digraph)
