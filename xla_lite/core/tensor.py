from __future__ import annotations

import copy
from dataclasses import dataclass, field

Data = int | float | list["Data"]


@dataclass
class Tensor:
    data: Data
    shape: tuple[int, ...] | None = field(init=False)

    def __post_init__(self):
        self.validate_tensor()
        self.shape = self.compute_shape(self.data)

    def compute_shape(self, data: Data) -> tuple[int]:
        shape: list[int] = []
        while isinstance(data, list):
            shape.append(len(data))
            if len(data) == 0:
                break
            data = data[0]
        return tuple(shape)

    def validate_tensor(self) -> None:
        if not isinstance(self.data, (int, float, list)):
            raise TypeError(
                "Tensor data must be an int, float, or nested list."
            )

        def check_uniform(data: Data, depth: int = 0) -> None:
            if isinstance(data, list):
                if len(data) == 0:
                    return
                first_elem = data[0]
                for elem in data:
                    if type(elem) != type(first_elem):
                        raise TypeError(
                            "All elements must be of the same type."
                        )
                    check_uniform(elem, depth + 1)

        check_uniform(self.data)

    def is_scalar(self) -> bool:
        return isinstance(self.data, (int, float))

    def is_vector(self) -> bool:
        return self.is_row_vector() or self.is_column_vector()

    def is_row_vector(self) -> bool:
        return self.is_matrix() and self.shape[0] == 1 and self.shape[1] >= 1

    def is_column_vector(self) -> bool:
        return self.is_matrix() and self.shape[1] == 1 and self.shape[0] >= 1

    def is_matrix(self) -> bool:
        return (
            isinstance(self.data, list)
            and len(self.data) > 0
            and all(isinstance(row, list) for row in self.data)
            and all(
                all(isinstance(elem, (int, float)) for elem in row)
                for row in self.data
            )
        )

    def copy(self) -> Tensor:
        return Tensor(copy.deepcopy(self.data))
