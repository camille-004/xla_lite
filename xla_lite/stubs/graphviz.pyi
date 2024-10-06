from typing import Any

class Digraph:
    def __init__(self, comment: str | None = None) -> None: ...
    def attr(
        self, rankdir: str | None = None, size: str | None = None
    ) -> None: ...
    def node(
        self, name: str, label: str, shape: str | None = None
    ) -> None: ...
    def edge(self, tail_name: str, head_name: str) -> None: ...
    def render(
        self,
        filename: str | None = None,
        view: bool = False,
        cleanup: bool = False,
    ) -> Any: ...
