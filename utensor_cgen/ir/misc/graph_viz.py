from random import random, seed

import matplotlib.pyplot as plt
from graphviz import Digraph
from matplotlib import cm as _cm
from utensor_cgen.logger import logger

plt.style.use('ggplot')


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


def viz_memalloc(ugraph, out_fname=None, fontsize=12, lw=15, cmap=_cm.BrBG_r, rand_seed=1024):
  seed(rand_seed)
  fig = plt.gcf()
  if TensorAllocationTransformer.KWARGS_NAMESCOPE not in ugraph.attributes:
    logger.info('No tensor allocation plan to visualize: %s', ugraph.name)
    return fig
  alloc_plan = ugraph.attributes[TensorAllocationTransformer.KWARGS_NAMESCOPE]
  topo_tensors = []
  visited_tensors = set()
  for op_name in ugraph.topo_order:
    op_info = ugraph.ops_info[op_name]
    for tensor in op_info.input_tensors:
      if tensor not in visited_tensors:
        topo_tensors.append(tensor)
        visited_tensors.add(tensor)
  for tensor in ugraph.output_tensors:
    if tensor not in visited_tensors:
      topo_tensors.append(tensor)
      visited_tensors.add(tensor)
  num_tensors = len(topo_tensors)
  ys = [num_tensors - i for i in range(len(topo_tensors))]
  xmins = [alloc_plan.plan[tensor.name].start for tensor in topo_tensors]
  xmaxs = [alloc_plan.plan[tensor.name].end for tensor in topo_tensors]
  colors = [cmap(random()) for _ in alloc_plan.plan]
  labels = [tensor.name for tensor in topo_tensors]
  sizes = [alloc_plan.plan[tensor.name].size for tensor in topo_tensors]
  for y, xmin, xmax, color, size in zip(ys, xmins, xmaxs, colors, sizes):
    plt.hlines(y, xmin, xmax, lw=lw, colors=color)
    plt.text(xmax, y-0.15, '{} bytes'.format(size), fontdict={'fontsize': fontsize})
  plt.xlabel('Offset (bytes)', fontdict={'fontsize': fontsize})
  plt.yticks(ys, labels, fontsize=fontsize)
  plt.ylabel('Tensor Names (Topological Ordered, Top to Bottom)', fontdict={'fontsize':fontsize})
  fig.set_size_inches(len(ys), len(ys)*0.5)
  plt.tight_layout()
  if out_fname:
    logger.info('saving tensor mem allocation to %s', out_fname)
    fig.savefig(out_fname)
  return fig

# cyclic import
from utensor_cgen.transformer.mem_alloc import TensorAllocationTransformer # isort:skip
