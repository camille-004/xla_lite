import pytest

from xla_lite.core import Graph, Node, OpType, Tensor
from xla_lite.execution import Executor


def test_executor_scalar_addition() -> None:
    graph = Graph()

    tensor_a = Tensor(5)
    tensor_b = Tensor(10)

    node_a = Node(node_id="a", tensor=tensor_a)
    node_b = Node(node_id="b", tensor=tensor_b)

    add_node = Node(node_id="c", op=OpType.ADD.value, inputs=["a", "b"])

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(add_node)

    executor = Executor(graph)
    results = executor.execute()

    assert "c" in results
    assert results["c"].is_scalar()
    assert results["c"].data == 15


def test_executor_scalar_multiplication() -> None:
    graph = Graph()

    tensor_a = Tensor(4)
    tensor_b = Tensor(3)

    node_a = Node(node_id="a", tensor=tensor_a)
    node_b = Node(node_id="b", tensor=tensor_b)
    mul_node = Node(node_id="c", op=OpType.MULTIPLY.value, inputs=["a", "b"])

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(mul_node)

    executor = Executor(graph)
    results = executor.execute()

    assert "c" in results
    assert results["c"].is_scalar()
    assert results["c"].data == 12


def test_executor_matrix_multiplication() -> None:
    graph = Graph()

    tensor_a = Tensor([[1, 2], [3, 4]])
    tensor_b = Tensor([[5, 6], [7, 8]])

    node_a = Node(node_id="a", tensor=tensor_a)
    node_b = Node(node_id="b", tensor=tensor_b)
    matmul_node = Node(node_id="c", op=OpType.MATMUL.value, inputs=["a", "b"])

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(matmul_node)

    executor = Executor(graph)
    results = executor.execute()

    expected = [[19, 22], [43, 50]]
    assert "c" in results
    assert results["c"].is_matrix()
    assert results["c"].data == expected


def test_executor_sequential_ops() -> None:
    graph = Graph()

    tensor_a = Tensor(2)
    tensor_b = Tensor(3)
    tensor_c = Tensor(4)

    node_a = Node(node_id="a", tensor=tensor_a)
    node_b = Node(node_id="b", tensor=tensor_b)
    node_c = Node(node_id="c", tensor=tensor_c)

    # d = a + b
    add_node = Node(node_id="d", op=OpType.ADD.value, inputs=["a", "b"])

    # e = d * c
    mul_node = Node(node_id="e", op=OpType.MULTIPLY.value, inputs=["d", "c"])

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)
    graph.add_node(add_node)
    graph.add_node(mul_node)

    executor = Executor(graph)
    results = executor.execute()

    assert "d" in results
    assert results["d"].is_scalar()
    assert results["d"].data == 5

    assert "e" in results
    assert results["e"].is_scalar()
    assert results["e"].data == 20


def test_executor_branching_ops() -> None:
    """
    Example:
        a + b = c
        a * b = d
    """
    graph = Graph()

    tensor_a = Tensor(3)
    tensor_b = Tensor(4)

    node_a = Node(node_id="a", tensor=tensor_a)
    node_b = Node(node_id="b", tensor=tensor_b)

    # c = a + b
    add_node = Node(node_id="c", op=OpType.ADD.value, inputs=["a", "b"])

    # d = a * b
    mul_node = Node(node_id="d", op=OpType.MULTIPLY.value, inputs=["a", "b"])

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(add_node)
    graph.add_node(mul_node)

    executor = Executor(graph)
    results = executor.execute()

    assert "c" in results
    assert results["c"].is_scalar()
    assert results["c"].data == 7

    assert "d" in results
    assert results["d"].is_scalar()
    assert results["d"].data == 12


def test_executor_multiple_dependencies() -> None:
    """
    Example:
        a + b = c
        c * d = e
        b * d = f
        e + f = g
    """
    graph = Graph()

    tensor_a = Tensor(1)
    tensor_b = Tensor(2)
    tensor_d = Tensor(3)

    node_a = Node(node_id="a", tensor=tensor_a)
    node_b = Node(node_id="b", tensor=tensor_b)
    node_d = Node(node_id="d", tensor=tensor_d)

    add_node = Node(node_id="c", op=OpType.ADD.value, inputs=["a", "b"])
    mul_node_e = Node(node_id="e", op=OpType.MULTIPLY.value, inputs=["c", "d"])
    mul_node_f = Node(node_id="f", op=OpType.MULTIPLY.value, inputs=["b", "d"])
    add_node_g = Node(node_id="g", op=OpType.ADD.value, inputs=["e", "f"])

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_d)
    graph.add_node(add_node)
    graph.add_node(mul_node_e)
    graph.add_node(mul_node_f)
    graph.add_node(add_node_g)

    executor = Executor(graph)
    results = executor.execute()

    assert "c" in results and results["c"].data == 3
    assert "e" in results and results["e"].data == 9
    assert "f" in results and results["f"].data == 6
    assert "g" in results and results["g"].data == 15


def test_executor_multiple_outputs() -> None:
    """
    Example:
        a + b = c
        a * b = d
    """
    graph = Graph()

    tensor_a = Tensor(4)
    tensor_b = Tensor(5)

    node_a = Node(node_id="a", tensor=tensor_a)
    node_b = Node(node_id="b", tensor=tensor_b)

    add_node = Node(node_id="c", op=OpType.ADD.value, inputs=["a", "b"])
    mul_node = Node(node_id="d", op=OpType.MULTIPLY.value, inputs=["a", "b"])

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(add_node)
    graph.add_node(mul_node)

    executor = Executor(graph)
    results = executor.execute()

    assert "c" in results and results["c"].data == 9
    assert "d" in results and results["d"].data == 20


def test_executor_empty_graph() -> None:
    graph = Graph()

    executor = Executor(graph)
    results = executor.execute()

    assert results == {}


def test_executor_graph_with_cycles() -> None:
    graph = Graph()

    tensor_a = Tensor(1)

    node_a = Node(node_id="a", tensor=tensor_a)
    node_b = Node(
        node_id="b", op=OpType.ADD.value, inputs=["a", "c"]
    )  # b depends on c
    node_c = Node(
        node_id="c", op=OpType.MULTIPLY.value, inputs=["b", "c"]
    )  # c depends on b

    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)

    executor = Executor(graph)

    with pytest.raises(ValueError, match="Graph has cycles."):
        executor.execute()
