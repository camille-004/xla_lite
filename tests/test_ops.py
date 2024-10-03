import pytest

from xla_lite.core import Tensor
from xla_lite.core.ops import add, matmul, multiply


def test_add_scalars() -> None:
    tensor_a = Tensor(5)
    tensor_b = Tensor(10)
    result = add(tensor_a, tensor_b)
    assert result.is_scalar()
    assert result.data == 15
    assert result.shape == ()


def test_add_vectors() -> None:
    tensor_a = Tensor([[1, 2, 3]])  # Row vector
    tensor_b = Tensor([[4, 5, 6]])
    result = add(tensor_a, tensor_b)
    assert result.is_row_vector()
    assert result.data == [[5, 7, 9]]
    assert result.shape == (1, 3)

    tensor_c = Tensor([[1], [2], [3]])  # Column vector
    tensor_d = Tensor([[4], [5], [6]])
    result_col = add(tensor_c, tensor_d)
    assert result_col.is_column_vector()
    assert result_col.data == [[5], [7], [9]]
    assert result_col.shape == (3, 1)


def test_add_matrices() -> None:
    tensor_a = Tensor([[1, 2], [3, 4]])
    tensor_b = Tensor([[5, 6], [7, 8]])
    result = add(tensor_a, tensor_b)
    assert result.is_matrix()
    assert result.data == [[6, 8], [10, 12]]
    assert result.shape == (2, 2)


def test_add_scalar_vector() -> None:
    tensor_scalar = Tensor(10)
    tensor_vector = Tensor([[1, 2, 3]])  # Row vector
    result = add(tensor_scalar, tensor_vector)
    assert result.is_row_vector()
    assert result.data == [[11, 12, 13]]
    assert result.shape == (1, 3)

    tensor_vector_col = Tensor([[1], [2], [3]])  # Colummn vector
    result_col = add(tensor_scalar, tensor_vector_col)
    assert result_col.is_column_vector()
    assert result_col.data == [[11], [12], [13]]
    assert result_col.shape == (3, 1)


def test_add_vector_scalar() -> None:
    tensor_vector = Tensor([[1, 2, 3]])  # Row vector
    tensor_scalar = Tensor(10)
    result = add(tensor_vector, tensor_scalar)
    assert result.is_row_vector()
    assert result.data == [[11, 12, 13]]
    assert result.shape == (1, 3)

    tensor_vector_col = Tensor([[1], [2], [3]])  # Column vector
    result_col = add(tensor_vector_col, tensor_scalar)
    assert result_col.is_column_vector()
    assert result_col.data == [[11], [12], [13]]
    assert result_col.shape == (3, 1)


def test_add_scalar_matrix() -> None:
    tensor_scalar = Tensor(5)
    tensor_matrix = Tensor([[1, 2], [3, 4]])
    result = add(tensor_scalar, tensor_matrix)
    assert result.is_matrix()
    assert result.data == [[6, 7], [8, 9]]
    assert result.shape == (2, 2)


def test_add_matrix_scalar() -> None:
    tensor_matrix = Tensor([[1, 2], [3, 4]])
    tensor_scalar = Tensor(5)
    result = add(tensor_matrix, tensor_scalar)
    assert result.is_matrix()
    assert result.data == [[6, 7], [8, 9]]
    assert result.shape == (2, 2)


def test_add_invalid_types() -> None:
    tensor_a = Tensor([[1, 2], [3, 4]])
    tensor_b = Tensor([[1, 2, 3]])
    with pytest.raises(
        ValueError,
        match="Matrices must have the same dimensions for addition.",
    ):
        add(tensor_a, tensor_b)


def test_add_mismatched_shapes() -> None:
    tensor_a = Tensor([[1, 2, 3]])  # Shape: (1, 3)
    tensor_b = Tensor([[4, 5]])  # Shape: (1, 2)
    with pytest.raises(
        ValueError, match="Vectors must be of the same length for addition."
    ):
        add(tensor_a, tensor_b)

    tensor_c = Tensor([[1, 2], [3, 4], [5, 6]])
    tensor_d = Tensor([[7, 8], [9, 10]])
    with pytest.raises(
        ValueError,
        match="Matrices must have the same dimensions for addition.",
    ):
        add(tensor_c, tensor_d)


def test_multiply_scalars() -> None:
    tensor_a = Tensor(5)
    tensor_b = Tensor(10)
    result = multiply(tensor_a, tensor_b)
    assert result.is_scalar()
    assert result.data == 50
    assert result.shape == ()


def test_multiply_vectors() -> None:
    tensor_a = Tensor([[1, 2, 3]])  # Row vector
    tensor_b = Tensor([[4, 5, 6]])
    result = multiply(tensor_a, tensor_b)
    assert result.is_row_vector()
    assert result.data == [[4, 10, 18]]
    assert result.shape == (1, 3)

    tensor_c = Tensor([[1], [2], [3]])  # Column vector
    tensor_d = Tensor([[4], [5], [6]])
    result_col = multiply(tensor_c, tensor_d)
    assert result_col.is_column_vector()
    assert result_col.data == [[4], [10], [18]]
    assert result_col.shape == (3, 1)


def test_multiply_matrices_not_supported() -> None:
    tensor_a = Tensor([[1, 2], [3, 4]])
    tensor_b = Tensor([[5, 6], [7, 8]])
    with pytest.raises(
        TypeError,
        match="Multiplication not supported between these tensor types.",
    ):
        multiply(tensor_a, tensor_b)


def test_multiply_scalar_vector() -> None:
    tensor_scalar = Tensor(10)
    tensor_vector = Tensor([[1, 2, 3]])
    result = multiply(tensor_scalar, tensor_vector)
    assert result.is_vector()
    assert result.data == [[10, 20, 30]]
    assert result.shape == (1, 3)


def test_multiply_vector_sclaar() -> None:
    tensor_vector = Tensor([[1, 2, 3]])
    tensor_scalar = Tensor(10)
    result = multiply(tensor_vector, tensor_scalar)
    assert result.is_row_vector()
    assert result.data == [[10, 20, 30]]
    assert result.shape == (1, 3)

    tensor_vector_col = Tensor([[1], [2], [3]])  # Column vector
    result_col = multiply(tensor_scalar, tensor_vector_col)
    assert result_col.is_column_vector()
    assert result_col.data == [[10], [20], [30]]
    assert result_col.shape == (3, 1)


def test_multiply_scalar_matrix() -> None:
    tensor_scalar = Tensor(3)
    tensor_matrix = Tensor([[1, 2], [3, 4]])
    result = multiply(tensor_scalar, tensor_matrix)
    assert result.is_matrix()
    assert result.data == [[3, 6], [9, 12]]
    assert result.shape == (2, 2)


def test_multiply_matrix_scalar() -> None:
    tensor_matrix = Tensor([[1, 2], [3, 4]])
    tensor_scalar = Tensor(3)
    result = multiply(tensor_matrix, tensor_scalar)
    assert result.is_matrix()
    assert result.data == [[3, 6], [9, 12]]
    assert result.shape == (2, 2)


def test_multiply_invalid_types() -> None:
    tensor_a = Tensor([[1, 2], [3, 4]])
    tensor_b = Tensor([[1, 2, 3]])
    with pytest.raises(
        TypeError,
        match="Multiplication not supported between these tensor types.",
    ):
        multiply(tensor_a, tensor_b)


def test_multiply_mismatched_shapes() -> None:
    tensor_a = Tensor([[1, 2, 3]])  # Shape: (1, 3)
    tensor_b = Tensor([[4, 5]])  # Shape: (1, 2)
    with pytest.raises(
        ValueError,
        match="Vectors must be of the same length for multiplication.",
    ):
        multiply(tensor_a, tensor_b)


def test_matmul_valid_matrices() -> None:
    tensor_a = Tensor([[1, 2], [3, 4]])
    tensor_b = Tensor([[5, 6], [7, 8]])

    result = matmul(tensor_a, tensor_b)
    assert result.is_matrix()
    assert result.data == [[19, 22], [43, 50]]
    assert result.shape == (2, 2)


def test_matmul_matrix_column_vector() -> None:
    tensor_matrix = Tensor([[1, 2], [3, 4]])
    tensor_col_vector = Tensor([[5], [6]])
    result = matmul(tensor_matrix, tensor_col_vector)
    assert result.is_column_vector()
    assert result.data == [[17], [39]]
    assert result.shape == (2, 1)


def test_matmul_row_vector_matrix() -> None:
    tensor_row_vector = Tensor([[5, 6]])
    tensor_matrix = Tensor([[7, 8, 9], [10, 11, 12]])
    result = matmul(tensor_row_vector, tensor_matrix)
    assert result.is_matrix()
    assert result.data == [[95, 106, 117]]
    assert result.shape == (1, 3)


def test_matmul_invalid_types() -> None:
    tensor_a = Tensor([[1, 2, 3]])
    tensor_b = Tensor(10)
    with pytest.raises(
        TypeError,
        match="Matrix multiplication is only supported for matrices and "
        + "vectors with compatible dimensions.",
    ):
        matmul(tensor_a, tensor_b)


def test_matmul_incompatible_shapes() -> None:
    tensor_a = Tensor([[1, 2, 3], [4, 5, 6]])
    tensor_b = Tensor([[7, 8], [9, 10]])
    with pytest.raises(
        ValueError,
        match="Number of columns in the first matrix must equal number of "
        + "rows in the second matrix.",
    ):
        matmul(tensor_a, tensor_b)


def test_matmul_empty_matrices() -> None:
    tensor_a = Tensor([])
    tensor_b = Tensor([[1, 2], [3, 4]])
    with pytest.raises(
        TypeError,
        match="Matrix multiplication is only supported for matrices and "
        + "vectors with compatible dimensions.",
    ):
        matmul(tensor_a, tensor_b)


def test_matmul_single_element_matrices() -> None:
    tensor_a = Tensor([[2]])
    tensor_b = Tensor([[3]])
    result = matmul(tensor_a, tensor_b)
    assert result.is_matrix()
    assert result.data == [[6]]
    assert result.shape == (1, 1)
