from xla_lite.core import Tensor


def validate_tensor(tensor: Tensor) -> None:
    if not isinstance(tensor, Tensor):
        raise ValueError(f"Expected Tensor, got {type(tensor)}")
