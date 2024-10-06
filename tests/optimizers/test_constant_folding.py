import pytest

from xla_lite.core import Graph, Node, OpType, Tensor
from xla_lite.optimizers import ConstantFolding, Optimizer


@pytest.fixture
def graph() -> Graph:
    return Graph()


@pytest.fixture
def optimizer(graph: Graph) -> Optimizer:
    return Optimizer(graph)


@pytest.fixture
def strategy() -> ConstantFolding:
    return ConstantFolding()


def test_constant_folding_add(graph: Graph, strategy: ConstantFolding) -> None:
    const1 = Node("const1", tensor=Tensor(5), op=OpType.CONST.value)
    const2 = Node("const2", tensor=Tensor(3), op=OpType.CONST.value)
    add_node = Node("add", op=OpType.ADD.value, inputs=["const1", "const2"])

    graph.add_node(const1)
    graph.add_node(const2)
    graph.add_node(add_node)

    strategy.apply(graph)

    folded_node = graph.get_node("add")
    assert folded_node is not None
    assert folded_node.op == OpType.CONST.value
    assert folded_node.inputs == []
    assert folded_node.tensor is not None
    assert folded_node.tensor.data == 8


def test_complex_expression(graph: Graph, strategy: ConstantFolding) -> None:
    const1 = Node("const1", tensor=Tensor(5), op=OpType.CONST.value)
    const2 = Node("const2", tensor=Tensor(3), op=OpType.CONST.value)
    const3 = Node("const3", tensor=Tensor(2), op=OpType.CONST.value)
    add_node = Node("add", op=OpType.ADD.value, inputs=["const1", "const2"])
    mul_node = Node("mul", op=OpType.MULTIPLY.value, inputs=["add", "const3"])

    for node in [const1, const2, const3, add_node, mul_node]:
        graph.add_node(node)

    strategy.apply(graph)

    folded_node = graph.get_node("mul")
    assert folded_node is not None
    assert folded_node.op == OpType.CONST.value
    assert folded_node.tensor is not None
    assert folded_node.tensor.data == 16


def test_folding_with_floating_point(
    graph: Graph, strategy: ConstantFolding
) -> None:
    const1 = Node("const1", tensor=Tensor(5.5), op=OpType.CONST.value)
    const2 = Node("const2", tensor=Tensor(2.2), op=OpType.CONST.value)
    mul_node = Node(
        "mul", op=OpType.MULTIPLY.value, inputs=["const1", "const2"]
    )

    for node in [const1, const2, mul_node]:
        graph.add_node(node)

    strategy.apply(graph)

    folded_node = graph.get_node("mul")
    assert folded_node is not None
    assert folded_node.op == OpType.CONST.value
    assert folded_node.tensor is not None
    assert isinstance(folded_node.tensor.data, float)
    assert abs(folded_node.tensor.data - 12.1) < 1e-6


def test_constant_folding_multiply(
    graph: Graph, strategy: ConstantFolding
) -> None:
    const1 = Node("const1", tensor=Tensor(5), op=OpType.CONST.value)
    const2 = Node("const2", tensor=Tensor(3), op=OpType.CONST.value)
    mul_node = Node(
        "mul", op=OpType.MULTIPLY.value, inputs=["const1", "const2"]
    )

    graph.add_node(const1)
    graph.add_node(const2)
    graph.add_node(mul_node)

    strategy.apply(graph)

    folded_node = graph.get_node("mul")
    assert folded_node is not None
    assert folded_node.op == OpType.CONST.value
    assert folded_node.inputs == []
    assert folded_node.tensor is not None
    assert folded_node.tensor.data == 15


def test_constant_folding_with_non_constant_input(
    graph: Graph, strategy: ConstantFolding
) -> None:
    const1 = Node("const1", tensor=Tensor(5), op=OpType.CONST.value)
    some_input = Node("some_input", op=OpType.ADD.value, inputs=[])
    non_const = Node("non_const", op=OpType.ADD.value, inputs=["some_input"])
    add_node = Node("add", op=OpType.ADD.value, inputs=["const1", "non_const"])

    graph.add_node(const1)
    graph.add_node(some_input)
    graph.add_node(non_const)
    graph.add_node(add_node)

    strategy.apply(graph)

    some_input_node = graph.get_node("some_input")
    assert some_input_node is not None
    assert some_input_node.op == OpType.ADD.value
    assert some_input_node.inputs == []

    non_const_node = graph.get_node("non_const")
    assert non_const_node is not None
    assert non_const_node.op == OpType.ADD.value
    assert non_const_node.inputs == ["some_input"]

    add_node = graph.get_node("add")
    assert add_node is not None
    assert add_node.op == OpType.ADD.value
    assert add_node.inputs == ["const1", "non_const"]
    assert add_node.tensor is None

    const1_node = graph.get_node("const1")
    assert const1_node is not None
    assert const1_node.op == OpType.CONST.value
    assert const1_node.tensor is not None
    assert const1_node.tensor.data == 5


def test_folding_with_commutative_operations(
    graph: Graph, strategy: ConstantFolding
) -> None:
    const1 = Node("const1", tensor=Tensor(5), op=OpType.CONST.value)
    const2 = Node("const2", tensor=Tensor(3), op=OpType.CONST.value)
    add1 = Node("add1", op=OpType.ADD.value, inputs=["const1", "const2"])
    add2 = Node("add2", op=OpType.ADD.value, inputs=["const2", "const1"])

    for node in [const1, const2, add1, add2]:
        graph.add_node(node)

    strategy.apply(graph)

    folded_node1 = graph.get_node("add1")
    folded_node2 = graph.get_node("add2")
    assert folded_node1 is not None and folded_node2 is not None
    assert folded_node1.op == OpType.CONST.value
    assert folded_node2.op == OpType.CONST.value
    assert folded_node1.tensor is not None and folded_node2.tensor is not None
    assert folded_node1.tensor.data == folded_node2.tensor.data == 8
