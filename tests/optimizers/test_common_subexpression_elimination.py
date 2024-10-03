import pytest

from xla_lite.core import Graph, Node, OpType, Tensor
from xla_lite.optimizers import CommonSubexpressionElimination


@pytest.fixture
def graph() -> Graph:
    return Graph()


@pytest.fixture
def strategy() -> CommonSubexpressionElimination:
    return CommonSubexpressionElimination()


def test_simple_common_subexpression(
    graph: Graph, strategy: CommonSubexpressionElimination
) -> None:
    """
    Initial graph:
    a(5) ->
            add1 ->
    b(3)            mul
            add2 ->

    Expected result:
    a(5) ->
            add1 -> mul
    b(3) ->

    add2 should be replaced with a reference to add1.
    """
    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    add1 = Node("add1", op=OpType.ADD.value, inputs=["a", "b"])
    add2 = Node("add2", op=OpType.ADD.value, inputs=["a", "b"])
    mul = Node("mul", op=OpType.MULTIPLY.value, inputs=["add1", "add2"])

    for node in [a, b, add1, add2, mul]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("add2").inputs == ["add1"]
    assert graph.get_node("mul").inputs == ["add1", "add1"]


def test_multiple_common_subexpressions(
    graph: Graph, strategy: CommonSubexpressionElimination
) -> None:
    """
    Initial graph:
    a(5) ->
            add1 ->
    b(3) ->         mul
            add2
            add3 ->
    c(2) ->
            add4

    Expected result:
    a(5) ->
            add1 ->
    b(3) ->         mul
            add3 ->
    c(2) ->

    add2 should be replaced with a reference to add1, and add4 with add3.
    """
    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    c = Node("c", tensor=Tensor(2), op=OpType.CONST.value)
    add1 = Node("add1", op=OpType.ADD.value, inputs=["a", "b"])
    add2 = Node("add2", op=OpType.ADD.value, inputs=["a", "b"])
    add3 = Node("add3", op=OpType.ADD.value, inputs=["b", "c"])
    add4 = Node("add4", op=OpType.ADD.value, inputs=["b", "c"])
    mul = Node("mul", op=OpType.MULTIPLY.value, inputs=["add1", "add3"])

    for node in [a, b, c, add1, add2, add3, add4, mul]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("add2").inputs == ["add1"]
    assert graph.get_node("add4").inputs == ["add3"]
    assert graph.get_node("mul").inputs == ["add1", "add3"]


def test_nested_common_subexpressions(
    graph: Graph, strategy: CommonSubexpressionElimination
) -> None:
    """
    Initial graph:
    a(5) ->
            add1 -> mul1 ->
    b(3) ->                 result
                    mul2 ->

    Expected result:
    a(5) ->
            add1 -> mul1 -> result
    b(3) ->

    mul2 should be replaced with a reference to mul1.
    """
    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    add1 = Node("add1", op=OpType.ADD.value, inputs=["a", "b"])
    mul1 = Node("mul1", op=OpType.MULTIPLY.value, inputs=["add1", "b"])
    mul2 = Node("mul2", op=OpType.MULTIPLY.value, inputs=["add1", "b"])
    result = Node("result", op=OpType.ADD.value, inputs=["mul1", "mul2"])

    for node in [a, b, add1, mul1, mul2, result]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("mul2").inputs == ["mul1"]
    assert graph.get_node("result").inputs == ["mul1", "mul1"]


def test_no_common_subexpressions(
    graph: Graph, strategy: CommonSubexpressionElimination
) -> None:
    """
    Initial graph:
    a(5) ->
            add ->
    b(3) ->         result
            mul ->
    c(2) ->

    No changes expected.
    """
    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    c = Node("c", tensor=Tensor(2), op=OpType.CONST.value)
    add = Node("add", op=OpType.ADD.value, inputs=["a", "b"])
    mul = Node("mul", op=OpType.MULTIPLY.value, inputs=["b", "c"])
    result = Node("result", op=OpType.SUBTRACT.value, inputs=["add", "mul"])

    for node in [a, b, c, add, mul, result]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("add").inputs == ["a", "b"]
    assert graph.get_node("mul").inputs == ["b", "c"]
    assert graph.get_node("result").inputs == ["add", "mul"]


def test_common_subexpression_with_constants(
    graph: Graph, strategy: CommonSubexpressionElimination
) -> None:
    """
    Initial graph:
    a(5) ->
            add1
    b(3) ->
            add2

    Expected result:
    a(5) ->
            add1
    b(3) ->

    add2 should be replaced with a constant reference to add1.
    """
    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    add1 = Node("add1", op=OpType.ADD.value, inputs=["a", "b"])
    add2 = Node("add2", op=OpType.ADD.value, inputs=["a", "b"])

    for node in [a, b, add1, add2]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("add2").op == OpType.CONST.value
    assert graph.get_node("add2").inputs == ["add1"]


def test_common_subexpression_with_commutative_operations(
    graph: Graph, strategy: CommonSubexpressionElimination
) -> None:
    """
    Initial graph:
    a(5) ->
            add1 ->
    b(3)            mul
            add2 ->

    Expected result:
    a(5) ->
            add1 -> mul
    b(3) ->

    add2 should be repaced with a reference to add1, despite different input
    order.
    """
    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    add1 = Node("add1", op=OpType.ADD.value, inputs=["a", "b"])
    add2 = Node("add2", op=OpType.ADD.value, inputs=["b", "a"])
    mul = Node("mul", op=OpType.MULTIPLY.value, inputs=["add1", "add2"])

    for node in [a, b, add1, add2, mul]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("add2").inputs == ["add1"]
    assert graph.get_node("mul").inputs == ["add1", "add1"]


def test_common_subexpression_across_different_operations(
    graph: Graph, strategy: CommonSubexpressionElimination
) -> None:
    """
    Initial graph:
    a(5) -> add
    b(3) -> mul

    No changes expected.
    """
    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    add = Node("add", op=OpType.ADD.value, inputs=["a", "b"])
    mul = Node("mul", op=OpType.MULTIPLY.value, inputs=["a", "b"])

    for node in [a, b, add, mul]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("add").inputs == ["a", "b"]
    assert graph.get_node("mul").inputs == ["a", "b"]


def test_common_subexpression_with_multiple_uses(
    graph: Graph, strategy: CommonSubexpressionElimination
) -> None:
    """
    Initial graph:
    a(5) ->
            add1 ->
    b(3) -> add2 -> result
            add3 ->

    Expected result:
    a(5) ->
            add1 -> result
    b(3) ->

    add2 and add3 should be replaced with references to add1.
    """
    a = Node("a", tensor=Tensor(5), op=OpType.CONST.value)
    b = Node("b", tensor=Tensor(3), op=OpType.CONST.value)
    add1 = Node("add1", op=OpType.ADD.value, inputs=["a", "b"])
    add2 = Node("add2", op=OpType.ADD.value, inputs=["a", "b"])
    add3 = Node("add3", op=OpType.ADD.value, inputs=["a", "b"])
    result = Node(
        "result", op=OpType.MULTIPLY.value, inputs=["add1", "add2", "add3"]
    )

    for node in [a, b, add1, add2, add3, result]:
        graph.add_node(node)

    strategy.apply(graph)

    assert graph.get_node("add2").inputs == ["add1"]
    assert graph.get_node("add3").inputs == ["add1"]
    assert graph.get_node("result").inputs == ["add1", "add1", "add1"]
