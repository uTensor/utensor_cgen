from graphviz import Digraph

from utensor_cgen.logger import logger


def viz_graph(ugraph, out_fname=None, view=False, cleanup=True, colored_nodes=None, suffix=''):
  if colored_nodes is None:
    colored_nodes = set()
  else:
    colored_nodes = set(colored_nodes)
  dot = Digraph()
  nodes = {}
  i = 0
  for node in ugraph.ops:
    nodes[node.name] = '{}_{}'.format(chr(ord('a') + i), suffix)
    colored = node.name in colored_nodes
    if colored:
      dot.node(
        nodes[node.name],
        "%s: %s" % (node.name, node.op_type),
        color='lightskyblue1',
        style='filled'
      )
    else:
      dot.node(nodes[node.name], "%s: %s" % (node.name, node.op_type))
    i += 1
    for n in node.input_tensors:
      if n.name in nodes:
        continue
      nodes[n.name] = '{}_{}'.format(chr(ord('a') + i), suffix)
      colored = n.name in colored_nodes
      if colored:
        dot.node(
          nodes[n.name],
          "%s: Tensor" % n.name,
          color='olivedrab2',
          style='filled',
          shape='box',
        )
      else:
        dot.node(nodes[n.name], "%s: Tensor" % n.name, shape='box')
      i += 1
    for n in node.output_tensors:
      if n.name in nodes:
        continue
      nodes[n.name] = '{}_{}'.format(chr(ord('a') + i), suffix)
      colored = n.name in colored_nodes
      if colored:
        dot.node(
          nodes[n.name],
          "%s: Tensor" % n.name,
          color='olivedrab2',
          style='filled',
          shape='box',
        )
      else:
        dot.node(nodes[n.name], "%s: Tensor" % n.name, shape='box')
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
