from graphviz import Digraph  # type: ignore

from xla_lite.core import Graph, Node, OpType


def visualize_graph(graph: Graph, filename: str | None = None) -> Digraph:
    dot = Digraph(comment="Computational Graph")
    dot.attr(rankdir="TB", size="8,8")

    def _add_node(node: Node):
        if node.op == OpType.CONST.value:
            label = f"{node.node_id}\n{node.tensor}"
            shape = "box"
        else:
            label = f"{node.node_id}\n{node.op}"
            shape = "ellipse"

        dot.node(node.node_id, label, shape=shape)

    def _add_edges(node: Node):
        for input_id in node.inputs:
            dot.edge(input_id, node.node_id)

    for node in graph.nodes:
        _add_node(node)
        _add_edges(node)

    if filename:
        dot.render(filename, view=True, cleanup=True)

    return dot


def add_viz_method(Graph):
    def visualize(self, filename: str | None = None) -> Digraph:
        return visualize_graph(self, filename)

    setattr(Graph, "visualize", visualize)


add_viz_method(Graph)
