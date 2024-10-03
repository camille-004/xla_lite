from xla_lite.core import Graph, Node, OpType
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
        return all(
            graph.get_node(input_id) is not None
            and graph.get_node(input_id).op == OpType.CONST.value
            for input_id in node.inputs
        )

    @staticmethod
    def _fold_node(graph: Graph, node: Node) -> None:
        inputs = [graph.get_node(input_id).tensor for input_id in node.inputs]
        folded = Optimizer._execute_operation(node.op, inputs)

        node.op = OpType.CONST.value
        node.inputs = []
        node.tensor = folded
