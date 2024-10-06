from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from typing import (
    Any,
    Callable,
    Literal,
    Sequence,
    TypeVar,
    cast,
    overload,
)

from xla_lite.core.tensor import Data, Tensor

T = TypeVar("T", bound=int | float | Sequence[Any])


class Operation(ABC):
    @abstractmethod
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        pass


class ElementWiseOperation(Operation):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        if a.is_scalar() and not b.is_scalar() and b.shape is not None:
            a = Tensor(
                self._broadcast_scalar(cast(int | float, a.data), b.shape)
            )
        elif b.is_scalar() and not a.is_scalar() and a.shape is not None:
            b = Tensor(
                self._broadcast_scalar(cast(int | float, b.data), a.shape)
            )

        if a.shape != b.shape:
            op_name = self.__class__.__name__.lower()
            if op_name in ("add", "subtract"):
                if a.is_matrix() and b.is_matrix():
                    raise ValueError(
                        "Matrices and vectors must have the same dimensions "
                        + "for addition or subtraction."
                    )
                elif a.is_vector() and b.is_vector():
                    raise ValueError(
                        "Vectors must be of the same length for addition or "
                        + "subtraction."
                    )
            else:
                raise ValueError(
                    f"Incompatible shapes for {op_name}: {a.shape} and "
                    + f"{b.shape}."
                )
        return self.operate(a, b)

    @abstractmethod
    def operate(self, a: Tensor, b: Tensor) -> Tensor:
        pass

    def _broadcast_scalar(
        self, scalar: int | float, shape: tuple[int, ...]
    ) -> Data:
        if len(shape) == 1:
            return [scalar] * shape[0]
        elif len(shape) == 2:
            return [[scalar] * shape[1] for _ in range(shape[0])]
        else:
            raise ValueError("Unsupported shape for broadcasting")

    @overload
    def element_wise_operation(
        self, a: Number, b: Number, op: Callable[[Number, Number], Number]
    ) -> Number: ...

    @overload
    def element_wise_operation(
        self,
        a: Sequence[Any],
        b: Sequence[Any],
        op: Callable[[Number, Number], Number],
    ) -> Sequence[Any]: ...

    def element_wise_operation(
        self,
        a: Number | Sequence[Any],
        b: Number | Sequence[Any],
        op: Callable[[Number, Number], Number],
    ) -> Number | Sequence[Any]:
        if isinstance(a, Number) and isinstance(b, Number):
            return op(a, b)
        elif isinstance(a, Sequence) and isinstance(b, Sequence):
            return [
                self.element_wise_operation(x, y, op) for x, y in zip(a, b)
            ]
        else:
            raise TypeError("Unsupported types for element-wise operation")


class Add(ElementWiseOperation):
    def operate(self, a: Tensor, b: Tensor) -> Tensor:
        result = self._element_wise_add(a.data, b.data)
        return Tensor(result)

    def _element_wise_add(self, a: Data, b: Data) -> Data:
        if isinstance(a, Number) and isinstance(b, Number):
            return a + b
        elif isinstance(a, Sequence) and isinstance(b, Sequence):
            return [self._element_wise_add(x, y) for x, y in zip(a, b)]
        else:
            raise TypeError("Unsupported types for addition")


class Subtract(ElementWiseOperation):
    def operate(self, a: Tensor, b: Tensor) -> Tensor:
        result = self._element_wise_subtract(a.data, b.data)
        return Tensor(result)

    def _element_wise_subtract(self, a: Data, b: Data) -> Data:
        if isinstance(a, Number) and isinstance(b, Number):
            return a - b
        elif isinstance(a, Sequence) and isinstance(b, Sequence):
            return [self._element_wise_subtract(x, y) for x, y in zip(a, b)]
        else:
            raise TypeError("Unsupported types for subtraction")


class Multiply(ElementWiseOperation):
    def operate(self, a: Tensor, b: Tensor) -> Tensor:
        result = self._element_wise_multiply(a.data, b.data)
        return Tensor(result)

    def _element_wise_multiply(self, a: Data, b: Data) -> Data:
        if isinstance(a, Number) and isinstance(b, Number):
            return a * b
        elif isinstance(a, Sequence) and isinstance(b, Sequence):
            return [self._element_wise_multiply(x, y) for x, y in zip(a, b)]
        else:
            raise TypeError("Unsupported types for multiplication")


class Divide(ElementWiseOperation):
    def operate(self, a: Tensor, b: Tensor) -> Tensor:
        result = self._element_wise_divide(a.data, b.data)
        return Tensor(result)

    def _element_wise_divide(self, a: Data, b: Data) -> Data:
        if isinstance(a, Number) and isinstance(b, Number):
            return a / b if b != 0 else float("inf")
        elif isinstance(a, Sequence) and isinstance(b, Sequence):
            return [self._element_wise_divide(x, y) for x, y in zip(a, b)]
        else:
            raise TypeError("Unsupported types for division")


class MatrixMultiply(Operation):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        if a.shape is None or b.shape is None:
            raise ValueError(
                "Tensors must have defined shapes for matrix multiplication."
            )

        if a.is_matrix() and b.is_matrix():
            return self.matrix_multiply(a, b)
        elif a.is_matrix() and b.is_vector():
            return self.matrix_vector_multiply(a, b)
        elif a.is_vector() and b.is_matrix():
            return self.vector_matrix_multiply(a, b)
        else:
            raise TypeError(
                "Matrix multiplication is only supported for matrices and "
                + "vectors with compatible dimensions."
            )

    def matrix_multiply(self, a: Tensor, b: Tensor) -> Tensor:
        assert a.shape is not None and b.shape is not None
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                "Number of columns in the first matrix must equal number of "
                + "rows in the second matrix."
            )

        result: list[list[float]] = []
        for i in range(a.shape[0]):
            row: list[float] = []
            for j in range(b.shape[1]):
                sum_product = sum(
                    float(cast(Sequence[Any], a.data)[i][k])
                    * float(cast(Sequence[Any], b.data)[k][j])
                    for k in range(a.shape[1])
                )
                row.append(sum_product)
            result.append(row)

        return Tensor(cast(Data, result))

    def matrix_vector_multiply(self, a: Tensor, b: Tensor) -> Tensor:
        assert a.shape is not None and b.shape is not None
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                "Length of the vector must equal number of rows in the matrix "
                + "for multiplication."
            )

        result = [
            sum(
                float(cast(Sequence[Any], a.data)[i][j])
                * float(cast(Sequence[Any], b.data)[j][0])
                for j in range(a.shape[1])
            )
            for i in range(a.shape[0])
        ]
        return Tensor(cast(Data, result))

    def vector_matrix_multiply(self, a: Tensor, b: Tensor) -> Tensor:
        assert a.shape is not None and b.shape is not None
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                "Length of the vector must equal number of rows in the matrix "
                + "for multiplication."
            )

        result = [
            sum(
                float(cast(Sequence[Any], a.data)[0][i])
                * float(cast(Sequence[Any], b.data)[i][j])
                for i in range(b.shape[0])
            )
            for j in range(b.shape[1])
        ]
        return Tensor(cast(Data, [result]))


# Operation factory
def get_operation(
    op: Literal["add", "subtract", "multiply", "divide", "matmul"],
) -> Operation:
    operations: dict[str, Operation] = {
        "add": Add(),
        "subtract": Subtract(),
        "multiply": Multiply(),
        "divide": Divide(),
        "matmul": MatrixMultiply(),
    }
    return operations[op]


# Main function
def operate(
    a: Tensor,
    b: Tensor,
    op: Literal["add", "subtract", "multiply", "divide", "matmul"],
) -> Tensor:
    operation = get_operation(op)
    return operation(a, b)


# Convenience functions
def add(a: Tensor, b: Tensor) -> Tensor:
    return operate(a, b, "add")


def subtract(a: Tensor, b: Tensor) -> Tensor:
    return operate(a, b, "subtract")


def multiply(a: Tensor, b: Tensor) -> Tensor:
    return operate(a, b, "multiply")


def divide(a: Tensor, b: Tensor) -> Tensor:
    return operate(a, b, "divide")


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return operate(a, b, "matmul")
