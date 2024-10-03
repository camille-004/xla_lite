from xla_lite.core import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    if a.is_scalar() and b.is_scalar():
        return add_scalars(a, b)
    elif (a.is_scalar() and b.is_vector()) or (
        a.is_vector() and b.is_scalar()
    ):
        return add_scalar_and_vector(a, b)
    elif (a.is_scalar() and b.is_matrix()) or (
        a.is_matrix() and b.is_scalar()
    ):
        return add_scalar_and_matrix(a, b)
    elif a.is_vector() and b.is_vector():
        return add_vectors(a, b)
    elif a.is_matrix and b.is_matrix():
        return add_matrices(a, b)
    else:
        raise TypeError("Addition not supported between these tensor types.")


def add_scalars(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(data=a.data + b.data)


def add_scalar_and_vector(a: Tensor, b: Tensor) -> Tensor:
    scalar, vector = (a.data, b.data) if a.is_scalar() else (b.data, a.data)
    if a.is_row_vector() or b.is_row_vector():
        result = [[scalar + x for x in row] for row in vector]
    else:
        result = [[scalar + x[0]] for x in vector]
    return Tensor(data=result)


def add_scalar_and_matrix(a: Tensor, b: Tensor) -> Tensor:
    scalar, matrix = (a.data, b.data) if a.is_scalar() else (b.data, a.data)
    result = [[scalar + x for x in row] for row in matrix]
    return Tensor(data=result)


def add_vectors(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError("Vectors must be of the same length for addition.")
    if a.is_row_vector():
        result = [
            [x + y for x, y in zip(row_a, row_b)]
            for row_a, row_b in zip(a.data, b.data)
        ]
    else:
        result = [[x[0] + y[0]] for x, y in zip(a.data, b.data)]
    return Tensor(data=result)


def add_matrices(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            "Matrices must have the same dimensions for addition."
        )
    result = [
        [x + y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a.data, b.data)
    ]
    return Tensor(data=result)


def subtract(a: Tensor, b: Tensor) -> Tensor:
    if a.is_scalar() and b.is_scalar():
        return subtract_scalars(a, b)
    elif (a.is_scalar() and b.is_vector()) or (
        a.is_vector() and b.is_scalar()
    ):
        return subtract_scalar_and_vector(a, b)
    elif (a.is_scalar() and b.is_matrix()) or (
        a.is_matrix() and b.is_scalar()
    ):
        return subtract_scalar_and_matrix(a, b)
    elif a.is_vector() and b.is_vector():
        return subtract_vectors(a, b)
    elif a.is_matrix and b.is_matrix():
        return subtract_matrices(a, b)
    else:
        raise TypeError(
            "Subtraction not supported between these tensor types."
        )


def subtract_scalars(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(data=a.data - b.data)


def subtract_scalar_and_vector(a: Tensor, b: Tensor) -> Tensor:
    scalar, vector = (a.data, b.data) if a.is_scalar() else (b.data, a.data)
    if a.is_row_vector() or b.is_row_vector():
        result = [[scalar - x for x in row] for row in vector]
    else:
        result = [[scalar - x[0]] for x in vector]
    return Tensor(data=result)


def subtract_scalar_and_matrix(a: Tensor, b: Tensor) -> Tensor:
    scalar, matrix = (a.data, b.data) if a.is_scalar() else (b.data, a.data)
    result = [[scalar - x for x in row] for row in matrix]
    return Tensor(data=result)


def subtract_vectors(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError("Vectors must be of the same length for subtraction.")
    if a.is_row_vector():
        result = [
            [x - y for x, y in zip(row_a, row_b)]
            for row_a, row_b in zip(a.data, b.data)
        ]
    else:
        result = [[x[0] - y[0]] for x, y in zip(a.data, b.data)]
    return Tensor(data=result)


def subtract_matrices(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            "Matrices must have the same dimensions for subtraction."
        )
    result = [
        [x - y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a.data, b.data)
    ]
    return Tensor(data=result)


def multiply(a: Tensor, b: Tensor) -> Tensor:
    if a.is_scalar() and b.is_scalar():
        return Tensor(data=a.data * b.data)

    elif (a.is_scalar() and b.is_vector()) or (
        a.is_vector() and b.is_scalar()
    ):
        if a.is_scalar():
            scalar = a.data
            vector = b.data
        else:
            scalar = b.data
            vector = a.data
        if a.is_row_vector() or b.is_row_vector():
            result = [[scalar * x for x in row] for row in vector]
        else:
            result = [[scalar * x[0]] for x in vector]
        return Tensor(data=result)

    elif (a.is_scalar() and b.is_matrix()) or (
        a.is_matrix() and b.is_scalar()
    ):
        if a.is_scalar():
            scalar = a.data
            matrix = b.data
        else:
            scalar = b.data
            matrix = a.data
        result = [[scalar * x for x in row] for row in matrix]
        return Tensor(data=result)

    elif a.is_vector() and b.is_vector():
        if a.shape != b.shape:
            raise ValueError(
                "Vectors must be of the same length for multiplication."
            )
        if a.is_row_vector():
            result = [
                [x * y for x, y in zip(row_a, row_b)]
                for row_a, row_b in zip(a.data, b.data)
            ]
        else:
            result = [[x[0] * y[0]] for x, y in zip(a.data, b.data)]
        return Tensor(data=result)

    else:
        raise TypeError(
            "Multiplication not supported between these tensor types."
        )


def divide(a: Tensor, b: Tensor) -> Tensor:
    if a.is_scalar() and b.is_scalar():
        return divide_scalars(a, b)
    elif a.is_scalar() and b.is_vector():
        return divide_scalar_and_vector(a, b)
    elif a.is_vector() and b.is_scalar():
        return divide_vector_and_scalar(a, b)
    elif a.is_scalar() and b.is_matrix():
        return divide_scalar_and_matrix(a, b)
    elif a.is_matrix() and b.is_scalar():
        return divide_matrix_and_scalar(a, b)
    elif a.is_vector() and b.is_vector():
        return divide_vectors(a, b)
    else:
        raise TypeError("Division not supported between these tensor types.")


def divide_scalars(a: Tensor, b: Tensor) -> Tensor:
    if b.data == 0:
        raise ValueError("Division by zero.")
    return Tensor(data=a.data / b.data)


def divide_scalar_and_vector(a: Tensor, b: Tensor) -> Tensor:
    if any(x == 0 for row in b.data for x in row):
        raise ValueError("Division by zero in vector.")
    result = [[a.data / x for x in row] for row in b.data]
    return Tensor(data=result)


def divide_vector_and_scalar(a: Tensor, b: Tensor) -> Tensor:
    if b.data == 0:
        raise ValueError("Division by zero.")
    result = [[x / b.data for x in row] for row in a.data]
    return Tensor(data=result)


def divide_scalar_and_matrix(a: Tensor, b: Tensor) -> Tensor:
    if any(x == 0 for row in b.data for x in row):
        raise ValueError("Division by zero in matrix.")
    result = [[a.data / x for x in row] for row in b.data]
    return Tensor(data=result)


def divide_matrix_and_scalar(a: Tensor, b: Tensor) -> Tensor:
    if b.data == 0:
        raise ValueError("Division by zero.")
    result = [[x / b.data for x in row] for row in a.data]
    return Tensor(data=result)


def divide_vectors(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(
            "Vectors must be of the same length for element-wise division."
        )
    if any(x == 0 for row in b.data for x in row):
        raise ValueError("Division by zero in vector.")
    result = [
        [x / y for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a.data, b.data)
    ]
    return Tensor(data=result)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if a.is_matrix() and b.is_matrix():
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                "Number of columns in the first matrix must equal number of "
                + "rows in the second matrix."
            )

        result = []
        for i in range(a.shape[0]):
            row = []
            for j in range(b.shape[1]):
                sum_product = 0
                for k in range(a.shape[1]):
                    sum_product += a.data[i][k] * b.data[k][j]
                row.append(sum_product)
            result.append(row)

        return Tensor(data=result)
    elif a.is_matrix() and b.is_vector():
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                "Length of the vector must equal number of rows in the matrix "
                + "for multiplication."
            )

        result = []
        for j in range(b.shape[1]):
            sum_product = 0
            for i in range(len(a.data)):
                sum_product += a.data[i] * b.data[i][j]
            result.append(sum_product)

        return Tensor(data=result)
    else:
        raise TypeError(
            "Matrix multiplication is only supported for matrices and vectors "
            + "with compatible dimensions."
        )
