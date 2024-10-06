import pytest

from xla_lite.core import Tensor


def test_tensor_scalar_creation() -> None:
    tensor = Tensor(5)
    assert tensor.is_scalar()
    assert tensor.shape == ()
    assert tensor.data == 5

    tensor_float = Tensor(3.14)
    assert tensor_float.is_scalar()
    assert tensor_float.shape == ()
    assert tensor_float.data == 3.14


def test_tensor_vector_creation() -> None:
    tensor_row = Tensor([[1, 2, 3]])
    assert tensor_row.is_vector()
    assert tensor_row.is_row_vector()
    assert not tensor_row.is_column_vector()

    tensor_col = Tensor([[1], [2], [3]])
    assert tensor_col.is_vector()
    assert tensor_col.is_column_vector()
    assert not tensor_col.is_row_vector()


def test_tensor_matrix_creation() -> None:
    tensor = Tensor([[1, 2], [3, 4]])
    assert tensor.is_matrix()
    assert tensor.shape == (2, 2)
    assert tensor.data == [[1, 2], [3, 4]]

    tensor_float = Tensor([[1.0, 2.0], [3.5, 4.5]])
    assert tensor_float.is_matrix()
    assert tensor_float.shape == (2, 2)
    assert tensor_float.data == [[1.0, 2.0], [3.5, 4.5]]


def test_tensor_invalid_creation() -> None:
    with pytest.raises(
        TypeError, match="Tensor data must be an int, float, or nested list."
    ):
        Tensor("invalid")  # type: ignore

    with pytest.raises(
        TypeError, match="All elements must be of the same type."
    ):
        Tensor([[1, "two", 3]])  # type: ignore

    with pytest.raises(
        TypeError, match="All elements must be of the same type."
    ):
        Tensor(data=[[1, 2], [3, "four"]])  # type: ignore
