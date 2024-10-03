import pytest

from xla_lite.core import Graph, Node, OpType, Tensor
from xla_lite.optimizers import DeadCodeElimination, Optimizer


@pytest.fixture
def graph() -> Graph:
    return Graph()


@pytest.fixture
def optimizer(graph: Graph) -> Optimizer:
    return Optimizer(graph)


@pytest.fixture
def strategy() -> DeadCodeElimination:
    return DeadCodeElimination()


def test_dead_code_elimination(
    graph: Graph, strategy: DeadCodeElimination
) -> None:
    input_node = Node("input", tensor=Tensor(5), op=OpType.CONST.value)
    dead_node = Node("dead", op=OpType.ADD.value, inputs=["input"])
    intermediate_node = Node(
        "intermediate", op=OpType.MULTIPLY.value, inputs=["input"]
    )
    output_node = Node(
        "output", OpType.MULTIPLY.value, inputs=["intermediate"]
    )
    output_node.is_output = True

    graph.add_node(input_node)
    graph.add_node(dead_node)
    graph.add_node(intermediate_node)
    graph.add_node(output_node)

    print("Graph before elimination:")
    print(graph, "\n")

    strategy.apply(graph)

    print("\nGraph after elimination:")
    print(graph)

    assert graph.get_node("dead") is None
    assert graph.get_node("input") is not None
    assert graph.get_node("intermediate") is not None
    assert graph.get_node("output") is not None


def test_no_dead_code(graph: Graph, strategy: DeadCodeElimination) -> None:
    input_node = Node("input", tensor=Tensor(5), op=OpType.CONST.value)
    intermediate_node = Node(
        "intermediate", op=OpType.ADD.value, inputs=["input"]
    )
    output_node = Node(
        "output", op=OpType.MULTIPLY.value, inputs=["intermediate"]
    )
    output_node.is_output = True

    graph.add_node(input_node)
    graph.add_node(intermediate_node)
    graph.add_node(output_node)

    print("Graph before elimination:")
    print(graph, "\n")

    strategy.apply(graph)

    print("\nGraph after elimination:")
    print(graph)

    assert graph.get_node("input") is not None
    assert graph.get_node("intermediate") is not None
    assert graph.get_node("output") is not None


def test_multiple_dead_nodes(
    graph: Graph, strategy: DeadCodeElimination
) -> None:
    input_node = Node("input", tensor=Tensor(5), op=OpType.CONST.value)
    dead_node1 = Node("dead1", op=OpType.ADD.value, inputs=["input"])
    dead_node2 = Node("dead2", op=OpType.MULTIPLY.value, inputs=["dead1"])
    output_node = Node("output", op=OpType.MULTIPLY.value, inputs=["input"])
    output_node.is_output = True

    for node in [input_node, dead_node1, dead_node2, output_node]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("dead1") is None
    assert graph.get_node("dead2") is None
    assert graph.get_node("input") is not None
    assert graph.get_node("output") is not None


def test_multiple_outputs(graph: Graph, strategy: DeadCodeElimination) -> None:
    input_node = Node("input", tensor=Tensor(5), op=OpType.CONST.value)
    intermediate = Node("intermediate", op=OpType.ADD.value, inputs=["input"])
    output1 = Node(
        "output1", op=OpType.MULTIPLY.value, inputs=["intermediate"]
    )
    output2 = Node("output2", op=OpType.SUBTRACT.value, inputs=["input"])
    output1.is_output = True
    output2.is_output = True

    for node in [input_node, intermediate, output1, output2]:
        graph.add_node(node)

    strategy.apply(graph)

    assert all(
        graph.get_node(node_id) is not None
        for node_id in ["input", "intermediate", "output1", "output2"]
    )


def test_branching_with_dead_branch(
    graph: Graph, strategy: DeadCodeElimination
) -> None:
    input_node = Node("input", tensor=Tensor(5), op=OpType.CONST.value)
    branch1 = Node("branch1", op=OpType.ADD.value, inputs=["input"])
    branch2 = Node("branch2", op=OpType.MULTIPLY.value, inputs=["input"])
    dead_end = Node("dead_end", op=OpType.SUBTRACT.value, inputs=["branch1"])
    output = Node("output", op=OpType.DIVIDE.value, inputs=["branch2"])
    output.is_output = True

    for node in [input_node, branch1, branch2, dead_end, output]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("dead_end") is None
    assert graph.get_node("branch1") is None
    assert all(
        graph.get_node(node_id) is not None
        for node_id in ["input", "branch2", "output"]
    )


def test_no_marked_outputs(
    graph: Graph, strategy: DeadCodeElimination
) -> None:
    input_node = Node("input", tensor=Tensor(5), op=OpType.CONST.value)
    intermediate = Node("intermediate", op=OpType.ADD.value, inputs=["input"])
    output = Node("output", op=OpType.MULTIPLY.value, inputs=["intermediate"])

    for node in [input_node, intermediate, output]:
        graph.add_node(node)

    strategy.apply(graph)

    assert all(
        graph.get_node(node_id) is not None
        for node_id in ["input", "intermediate", "output"]
    )


def test_circular_dependency(
    graph: Graph, strategy: DeadCodeElimination
) -> None:
    node1 = Node("node1", op=OpType.ADD.value, inputs=["node3"])
    node2 = Node("node2", op=OpType.MULTIPLY.value, inputs=["node1"])
    node3 = Node("node3", op=OpType.SUBTRACT.value, inputs=["node2"])
    output = Node("output", op=OpType.DIVIDE.value, inputs=["node2"])
    output.is_output = True

    for node in [node1, node2, node3, output]:
        graph.add_node(node)

    strategy.apply(graph)

    assert all(
        graph.get_node(node_id) is not None
        for node_id in ["node1", "node2", "node3", "output"]
    )


def test_isolated_nodes(graph: Graph, strategy: DeadCodeElimination) -> None:
    input1 = Node("input1", tensor=Tensor(5), op=OpType.CONST.value)
    input2 = Node("input2", tensor=Tensor(3), op=OpType.CONST.value)
    isolated = Node("isolated", tensor=Tensor(7), op=OpType.CONST.value)
    output = Node("output", op=OpType.ADD.value, inputs=["input1", "input2"])
    output.is_output = True

    for node in [input1, input2, isolated, output]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("isolated") is None
    assert all(
        graph.get_node(node_id) is not None
        for node_id in ["input1", "input2", "output"]
    )
