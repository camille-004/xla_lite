from xla_lite.core import Graph, Node, OpType, Tensor
from xla_lite.optimizers import Optimizer, OptStrategy


class ConstantFolding(OptStrategy):
    def apply(self, graph: Graph) -> None:
        changed = True
        while changed:
            changed = False
            for node in graph.nodes:
                if node.op and node.op != OpType.CONST.value:
                    if self._can_fold(graph, node):
                        self._fold_node(graph, node)
                        changed = True
                        print(f"Folded node {node.node_id}")

    @staticmethod
    def _can_fold(graph: Graph, node: Node) -> bool:
        return (
            all(
                input_node is not None and input_node.op == OpType.CONST.value
                for input_id in node.inputs
                if (input_node := graph.get_node(input_id)) is not None
            )
            and len(node.inputs) > 0
        )

    @staticmethod
    def _fold_node(graph: Graph, node: Node) -> None:
        inputs: list[Tensor] = []
        for input_id in node.inputs:
            input_node = graph.get_node(input_id)
            if input_node is None or input_node.tensor is None:
                raise ValueError(
                    f"Invalid input node or tensor for node {node.node_id}"
                )
            inputs.append(input_node.tensor)

        if node.op is None:
            raise ValueError(f"Node {node.node_id} has no operation")

        if len(inputs) < 2:
            return  # Don't fold if there are not enough inputs.

        folded = Optimizer._execute_operation(node.op, inputs)

        node.op = OpType.CONST.value
        node.inputs = []
        node.tensor = folded
