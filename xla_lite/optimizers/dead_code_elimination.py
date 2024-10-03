from collections import deque

from xla_lite.core import Graph
from xla_lite.optimizers import OptStrategy


class DeadCodeElimination(OptStrategy):
    def apply(self, graph: Graph) -> None:
        outputs = [
            node for node in graph.nodes if getattr(node, "is_output", False)
        ]
        if not outputs:
            print("No outputs found, considering the last node as output")
            outputs = [graph.nodes[-1]]

        print(
            f"Starting from output nodes: {[node.node_id for node in outputs]}"
        )

        reachable: set[str] = set()
        queue = deque(outputs)

        while queue:
            node = queue.popleft()
            if node.node_id in reachable:
                continue
            reachable.add(node.node_id)
            print(f"Marked node {node.node_id} as reachable")

            for input_id in node.inputs:
                input_node = graph.get_node(input_id)
                if input_node and input_node.node_id not in reachable:
                    queue.append(input_node)

        print(f"Reachable nodes: {reachable}")

        graph.nodes = [
            node for node in graph.nodes if node.node_id in reachable
        ]
        graph.node_map = {
            node_id: node
            for node_id, node in graph.node_map.items()
            if node_id in reachable
        }

        print(
            "Nodes after elimination: "
            + "{[node.node_id for node in graph.nodes]}"
        )
