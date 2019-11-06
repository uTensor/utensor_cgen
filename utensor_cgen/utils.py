# -*- coding: utf8 -*-
import importlib
import os
import re
import types
from ast import literal_eval
from collections import deque
from copy import deepcopy
from random import choice
from string import ascii_letters, digits

import numpy as np
from click.types import ParamType

import idx2numpy as idx2np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph
from utensor_cgen.logger import logger

__all__ = ["save_idx", "save_consts", "save_graph", "log_graph",
           "NamescopedKWArgsParser", "NArgsParam", "MUST_OVERWRITEN"]


def log_graph(graph_or_graph_def, logdir):
  if isinstance(graph_or_graph_def, tf.GraphDef):
    graph = tf.Graph()
    with graph.as_default():
      tf.import_graph_def(graph_or_graph_def, name='')
  else:
    graph = graph_or_graph_def
  tf.summary.FileWriter(logdir, graph=graph).close()


def save_idx(arr, fname):
  if arr.shape == ():
    arr = np.array([arr], dtype=arr.dtype)
  if arr.dtype in [np.int64]:
    logger.warning("unsupported int format for idx detected: %s, using int32 instead", arr.dtype)
    arr = arr.astype(np.int32)
  out_dir = os.path.dirname(fname)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir)
  with open(fname, "wb") as fid:
    idx2np.convert_to_file(fid, arr)
  logger.info("%s saved", fname)


def save_consts(sess, out_dir="."):
  out_dir = os.path.expanduser(out_dir)
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)
  graph = sess.graph
  graph_def = sess.graph.as_graph_def()
  for node in graph_def.node:
    if node.op == "Const":
      op = graph.get_operation_by_name(node.name)
      for out_tensor in op.outputs:
        arr = out_tensor.eval()
        tname = re.sub(r'[/:]', '_', out_tensor.name)
        idx_fname = os.path.join(out_dir, "{}.idx".format(tname))
        save_idx(arr, idx_fname)


def save_graph(graph, graph_name="graph", out_dir="."):
  out_dir = os.path.expanduser(out_dir)
  graph_fname = os.path.join(out_dir, "{}.pb".format(graph_name))
  with tf.gfile.FastGFile(graph_fname, "wb") as fid:
    fid.write(graph.as_graph_def().SerializeToString())
  logger.info("%s saved", graph_fname)


def prepare_meta_graph(meta_graph_path, output_nodes, chkp_path=None):
  """
  Cleanup and freeze the graph, given the path to meta graph data

  :param meta_graph_path: the path to the meta graph data (`.meta`)
  :type meta_graph_path: str
  :param output_nodes: a list of output node names in the graph
  :type output_nodes: List[str]
  :param chkp_path: the path of checkpoint directory
  :type chkp_path: str

  :rtype: tensorflow.GraphDef

  Basically, this function import the :class:`GraphDef` then:

    1. remove training nodes
    2. convert variable to constants
  """
  graph = tf.Graph()
  saver = tf.train.import_meta_graph(meta_graph_path,
                                     clear_devices=True,
                                     graph=graph)
  if chkp_path is None:
    chkp_path = meta_graph_path.replace(".meta", "")
  with tf.Session(graph=graph) as sess:
    saver.restore(sess, chkp_path)
    graph_def = graph_util.remove_training_nodes(sess.graph_def)
    sub_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                              input_graph_def=graph_def,
                                                              output_node_names=output_nodes)
  return sub_graph_def


def _sanitize_op_name(op_name):
  """
  Sanitize the op name
  - ignore '^' character of control input
  """
  if op_name.startswith('^'):
    return op_name[1:]
  return op_name


def parse_tensor_name(tname):
  """Adapt from TensorFlow source code
  tensor name --> (op_name, index)
  """
  components = tname.split(":")
  if len(components) == 2:
    op_name = _sanitize_op_name(components[0])
    try:
      output_index = int(components[1])
    except ValueError:
      raise ValueError("invalid output index: {}".format(tname))
    return (op_name, output_index)
  elif len(components) == 1:
    op_name = _sanitize_op_name(components[0])
    return (op_name, 0)
  else:
    raise ValueError("invalid tensor name: {}".format(tname))


class NamescopedKWArgsParser:
  """
  Find values under a namescope

  :param namespace: the target namespace
  :type namespace: str

  :param kwargs: a dict with key as str
  :type kwargs: dict

  Given a dict, ``kwargs``, this parer will parse
  its keys according to following rule:

  1. if the key match the pattern, r'^([^\d\W][\w\d_]*)__([^\d\W][\w\d_]*)',
    and the first matched group is the same as given ``namescope``, the value
    will be included as a private value under this namescope
  2. if the key do not match the pattern, then it's included as shared value
    among any namescope
  3. Otherwise, the value is excluded.

  See also :py:meth:`.NamescopedKWArgsParser.get` for examples

  TODO: replace it with a better data manager
  """
  def __init__(self, namespace, kwargs, data_manager=None, op_info=None):
    ns_pattern = re.compile(r'^([^\d\W][\w\d_]*)__([^\d\W][\w\d_]*)')
    self._namespace = namespace
    self._private_kwargs = {}
    self._shared_kwargs = {}
    for key, value in kwargs.items():
      match = ns_pattern.match(key)
      if match:
        ns = match.group(1)
        argname = match.group(2)
        if ns == self._namespace:
          self._private_kwargs[argname] = value
      else:
        self._shared_kwargs[key] = value
    if op_info is not None and data_manager is not None:
      outputs = [tensor_info.name for tensor_info in op_info.output_tensors]
      for tensor in outputs:
        values = data_manager.group(tensor)
        for key, value in values.items():
          if key not in self._private_kwargs:
            self._private_kwargs[key] = []
            self._private_kwargs[key].append(value)
          else:
            self._private_kwargs[key].append(value)
  
  def get(self, argname, default=None):
    """
    Get value of given name in the namespace

    This method mimic the behaviour of :py:meth:`dict.get`.
    If the given name can't be found in the namespace, ``default``
    is returned (default to ``None``)

    :param argname: the name of the value
    :type argname: str

    :param default: the default value to return for missing value

    .. code-block:: python

      kwargs = {
        'NS1__x': 1, 'NS2__x':2, x: 3, y:2
      }
      ns1_parser = NamescopedKWArgsParser('NS1', kwargs)
      ns2_parser = NamescopedKWArgsParser('NS2', kwargs)

      # the private value is returened and has higher priority
      # over shared value (``x`` in this case)
      ns1_parser.get('x')
      >>> 1
      ns2_parser.get('x')
      >>> 2
      ns1_parser.get('y')
      >>> 2
      ns1_parser.get('y') == ns2_parser.get('y')
      >>> True
    """
    try:
      return self._private_kwargs[argname]
    except KeyError:
      return self._shared_kwargs.get(argname, default)

  # def as_dict(self):
  #   kwargs = deepcopy(self._private_kwargs)
  #   for k, v in self._shared_kwargs.items():
  #     if not k in kwargs:
  #       kwargs[k] = v
  #   return kwargs

  def __repr__(self):
    d = dict(('%s__%s' % (self._namespace, k), v)
              for k, v in self._private_kwargs.items())
    repr_str = ('KWArgsParser(' +
                '%s, ' % self._namespace +
                '%s)' % d)
    return repr_str

  def __getitem__(self, argname):
    try:
      return self._private_kwargs[argname]
    except KeyError:
      return self._shared_kwargs[argname]


class NArgsParam(ParamType):
  """Click param type

  split given string value by seperator

  :param sep: seperator to split the string
  :type sep: str

  See also |click_param|_

  .. |click_param| replace:: `Click: Implementing Custom Types`
  .. _click_param: https://click.palletsprojects.com/en/7.x/parameters/#implementing-custom-types
  """

  def __init__(self, sep=','):
    self._sep = sep

  def convert(self, value, param, ctx):
    value = str(value)
    args = value.split(self._sep)
    aug_args = [arg for arg in args if arg[0] in ['+', '-']]
    if aug_args:
      final_args = param.default.split(self._sep)
      for arg in aug_args:
        if arg[0] == '+':
          final_args.append(arg[1:])
        elif arg[0] == '-' and arg[1:] in final_args:
          final_args.remove(arg[1:])
    else:
      final_args = args
    return final_args


class NArgsKwargsParam(NArgsParam):

  _trans_name_patrn = re.compile(r"(\w[\w]*)\(?")

  def convert(self, value, param, ctx):
    args = super(NArgsKwargsParam, self).convert(value, param, ctx)
    return [self._parse_kwargs(arg) for arg in args]
  
  def _parse_kwargs(self, arg):
    trans_match = self._trans_name_patrn.match(arg)
    if not trans_match:
      raise ValueError("Invalid args detected: {}".format(arg))
    trans_name = trans_match.group(1)
    _, end = trans_match.span()
    if end == len(arg):
      kwargs = {}
    else:
      if not arg.endswith(")"):
        raise ValueError("parentheses mismatch: {}".format(arg))
      kwargs = self._get_kwargs(arg[end:-1])
    return trans_name, kwargs
  
  def _get_kwargs(self, kws_str):
    kw_arg_strs = [s.strip() for s in kws_str.split(',')]
    kwargs = {}
    for kw_str in kw_arg_strs:
      name, v_str = kw_str.split('=')
      value = literal_eval(v_str)
      kwargs[name] = value
    return kwargs


class _MustOverwrite(object):
  _obj = None

  def __new__(cls, *args, **kwargs):
    if cls._obj is None:
      cls._obj = object.__new__(cls, *args, **kwargs)
    return cls._obj

MUST_OVERWRITEN = _MustOverwrite()


def topologic_order_graph(ugraph):
  """
  topological sort a given ugraph *in place*

  :param ugraph: the graph to be sorted
  :type ugraph: :class:`.uTensorGraph`

  - `Topological Sorting (wiki) <https://en.wikipedia.org/wiki/Topological_sorting>`_
  """
  ugraph.topo_order = get_topologic_order(ugraph, ugraph.output_nodes)[::-1]


def get_topologic_order(ugraph, init_nodes=None):
  """
  return list of op names in topological order

  :param ugraph: the graph to be sorted
  :type ugraph: :class:`.uTensorGraph`
  :param init_nodes: the initial nodes to start
    sorting with
  :type init_nodes: List[str]

  :rtype: List[str]

  - `Topological Sorting (wiki) <https://en.wikipedia.org/wiki/Topological_sorting>`_
  """
  if ugraph.backend != "tensorflow":
    raise ValueError(
      "topologic_order_graph works only on tensorflow graph"
    )
  if init_nodes is None:
    init_nodes = ugraph.output_nodes
  queue = deepcopy(init_nodes)
  visited = set()    # temporary mark
  perm_visit = set()  # Permanent mark
  ops_torder = []  # L

  def visit(node_name):
    if node_name in perm_visit:
      return
    if node_name in visited:
      raise ValueError("Input graph is not a DAG")
    visited.add(node_name)
    op_info = ugraph.ops_info.get(node_name, None)
    if not op_info:
      return

    for t_info in op_info.input_tensors:
      # NT: we should not rely on tensor-name conventions for back-tracing
      # op_name = parse_tensor_name(t_info.name)[0]
      # It would be nice to rely on something similar to get_tensor_node_names(), but based on ops_info instead of topo_order
      op_name = t_info.op_name
      visit(op_name)

    perm_visit.add(node_name)
    ops_torder.insert(0, node_name)

  while queue:
    node_name = queue.pop(0)
    visit(node_name)
  return ops_torder


def ops_bfs_queue(ugraph, init_nodes=None):
  if init_nodes is None:
    init_nodes = [
      ugraph.ops_info[name] for name in ugraph.output_nodes
    ]
  queue = deque(init_nodes)
  visited = set()
  bfs_deck = deque([])

  while queue:
    op = queue.popleft()
    if op is None or op.name in visited:
      # op is None => traversal via a null tensor
      # or been visited before
      continue
    visited.add(op.name)
    queue.extend(op.input_nodes)
    bfs_deck.append(op)
  return bfs_deck


def prune_graph(ugraph):
  """
  Remove nodes that is no longer needed *in-place*

  this function will trace the output nodes of the
  given graph by `BFS <https://en.wikipedia.org/wiki/Breadth-first_search>`_
  and remove all nodes which are not reachable afterward

  :param ugraph: the graph to be pruned
  :type ugraph: :class:`.uTensorGraph`
  """
  new_ugraph = deepcopy(ugraph)
  # BFS to find all ops you need
  ops_in_need = set(ugraph.output_nodes)
  queue = [name for name in ugraph.output_nodes]
  visited = set([])
  while queue:
    op_name = queue.pop(0)
    #TODO: move the code below to a standalone function.
    # Maybe using a more extensive data structure
    # or simply: in_ops = [node.name for node in ugraph.ops_info[op_name].input_nodes]
    tensors_in = set([t.name for t in ugraph.ops_info[op_name].input_tensors])
    in_ops = set()
    for it_node in ugraph.ops_info:
      if it_node == op_name:
        continue
      it_tensors_out = [t.name for t in ugraph.ops_info[it_node].output_tensors]
      if not tensors_in.isdisjoint(it_tensors_out):
        in_ops.add(it_node)

    queue.extend([name for name in in_ops if name not in visited])
    visited.update(in_ops)
    ops_in_need.update(in_ops)

  ops_to_remove = set([])
  for op_name in new_ugraph.ops_info.keys():
    if op_name not in ops_in_need:
      # remove ops not needed from ops_info
      ops_to_remove.add(op_name)
  for op_name in ops_to_remove:
    new_ugraph.ops_info.pop(op_name)
  return new_ugraph


def random_str(length=8):
  letters = ascii_letters+digits
  chars = [choice(letters) for _ in range(length)]
  return ''.join(chars)


class LazyLoader(types.ModuleType):

  def __init__(self, submod_name):
    self._submod_name = submod_name
    self._submod = None
    super(LazyLoader, self).__init__(submod_name)

  def _load(self):
    if self._submod is None:
      self._submod = importlib.import_module('utensor_cgen.{}'.format(self._submod_name))
    return self._submod

  def __getattr__(self, attrb):
    return getattr(self._load(), attrb)

  def __dir__(self):
    return dir(self._load())
