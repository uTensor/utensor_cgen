import os
from math import ceil, log10
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


class _MemallcVisualizer(object):

  @classmethod
  def viz_memalloc(
    cls,
    ugraph,
    out_fname=None,
    split_on_large_graph=True,
    num_tensors_per_split=20,
    figsize=None, 
    fontsize=12,
    lw=15, 
    cmap=_cm.BrBG_r,
    rand_seed=1111
  ):
    seed(rand_seed)
    if TensorAllocationTransformer.KWARGS_NAMESCOPE not in ugraph.attributes:
      logger.info('No tensor allocation plan to visualize: %s', ugraph.name)
      return plt.Figure()
    alloc_plan = ugraph.attributes[TensorAllocationTransformer.KWARGS_NAMESCOPE]
    topo_tensors = []
    visited_tensors = set()
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      for tensor in op_info.input_tensors:
        if tensor.name in alloc_plan.plan and tensor not in visited_tensors:
          topo_tensors.append(tensor)
          visited_tensors.add(tensor)
    for tensor in ugraph.output_tensors:
      if tensor.name in alloc_plan.plan and tensor not in visited_tensors:
        topo_tensors.append(tensor)
        visited_tensors.add(tensor)
    if out_fname:
      figs = cls._draw_figs(topo_tensors, alloc_plan, cmap, figsize, fontsize, lw, split_on_large_graph, num_tensors_per_split)
      if len(figs) == 1:
        logger.info('saving tensor mem allocation to %s', out_fname)
        figs[0].savefig(out_fname)
      else:
        num_digits = ceil(log10(len(figs)))
        file_format = '{{}}_{{:0{}d}}{{}}'.format(num_digits)
        for i, fig in enumerate(figs, 1):
          fname, ext = os.path.splitext(out_fname)
          fname = file_format.format(fname, i, ext)
          logger.info('saving tensor mem allocation to %s', fname)
          fig.savefig(fname)
  
  @classmethod
  def _draw_figs(cls, topo_tensors, alloc_plan, cmap, figsize, fontsize, lw, split_on_large_graph, num_tensors_per_split):
    xmins = [alloc_plan.plan[tensor.name].start for tensor in topo_tensors]
    xmaxs = [alloc_plan.plan[tensor.name].end for tensor in topo_tensors]
    colors = [cmap(random()) for _ in alloc_plan.plan]
    labels = [tensor.name for tensor in topo_tensors]
    sizes = [alloc_plan.plan[tensor.name].size for tensor in topo_tensors]
    if split_on_large_graph:
      xmin_chunks = [xmins[i:i+num_tensors_per_split] for i in range(0, len(xmins), num_tensors_per_split)]
      xmax_chunks = [xmaxs[i:i+num_tensors_per_split] for i in range(0, len(xmaxs), num_tensors_per_split)]
      color_chunks = [colors[i:i+num_tensors_per_split] for i in range(0, len(colors), num_tensors_per_split)]
      label_chunks = [labels[i:i+num_tensors_per_split] for i in range(0, len(labels), num_tensors_per_split)]
      size_chunks = [sizes[i:i+num_tensors_per_split] for i in range(0, len(sizes), num_tensors_per_split)]
    else:
      xmin_chunks = [xmins]
      xmax_chunks = [xmaxs]
      color_chunks = [colors]
      label_chunks = [labe]
      size_chunks = [sizes]
    figs = []
    for i, (xmins, xmaxs, colors, sizes) in enumerate(zip(xmin_chunks, xmax_chunks, color_chunks, size_chunks)):
      fig, _ = plt.subplots(1, 1)
      ys = [len(xmins)-i for i in range(len(xmins))]
      for y, xmin, xmax, color, size in zip(ys, xmins, xmaxs, colors, sizes):
        plt.hlines(y, xmin, xmax, lw=lw, colors=color)
        plt.text(xmax+lw*5, y-0.2*lw/3, '{} bytes'.format(size), fontdict={'fontsize': fontsize})
      plt.xlabel('Offset (bytes)', fontdict={'fontsize': fontsize})
      plt.yticks(ys, labels, fontsize=fontsize)
      plt.xticks(fontsize=fontsize)
      plt.ylabel(
        'Tensor Names (Topological Ordered, Top to Bottom)',
        fontdict={'fontsize':fontsize}
      )
      if i:
        title = 'Optimal Tensor Allocation: {} bytes in total (Cont.)'.format(alloc_plan.total_size)
      else:
        title = 'Optimal Tensor Allocation: {} bytes in total'.format(alloc_plan.total_size)
      plt.title(
        title,
        fontdict={'fontsize': fontsize}
      )
      if figsize is None:
        figsize = (num_tensors_per_split, num_tensors_per_split / 2)
      fig.set_size_inches(*figsize)
      fig.tight_layout()
      figs.append(fig)
    return figs

viz_memalloc = _MemallcVisualizer.viz_memalloc

# FIXME: cyclic import
from utensor_cgen.transformer.mem_alloc import TensorAllocationTransformer # isort:skip
