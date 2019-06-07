from graphviz import Digraph

from utensor_cgen.logger import logger


def viz_graph(ugraph, out_fname=None, view=False, cleanup=True):
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
  if out_fname:
    dot.render(out_fname, view=view, cleanup=cleanup)
    logger.info('graph visualization file generated: %s', out_fname)
  return dot
