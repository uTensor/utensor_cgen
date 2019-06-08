from utensor_cgen.ir import OperationInfo, uTensorGraph
from utensor_cgen.logger import logger
from graphviz import Digraph

def viz_graph(out_fname, view, ugraph):
    dot = Digraph()
    nodes = {}
    i = 0
    for node in ugraph.ops:
        nodes[node.name] = chr(ord('a') + i)
        dot.node(nodes[node.name], "%s: %s" % (node.name, node.op_type))
        i += 1
        for n in node.input_tensors:
            if n.name in nodes:
                continue
            nodes[n.name] = chr(ord('a') + i)
            dot.node(nodes[n.name], "%s: Tensor" % n.name)
            i += 1
        for n in node.output_tensors:
            if n.name in nodes:
                continue
            nodes[n.name] = chr(ord('a') + i)
            dot.node(nodes[n.name], "%s: Tensor" % n.name)
            i += 1
    for node in ugraph.ops:
        for n in node.input_tensors:
            dot.edge(nodes[n.name], nodes[node.name])
        for n in node.output_tensors:
            dot.edge(nodes[node.name], nodes[n.name])
    dot.render(out_fname, view=view)
    logger.info('graph visualization file generated: %s', out_fname)