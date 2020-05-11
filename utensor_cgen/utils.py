# -*- coding: utf8 -*-
import importlib
import os
import re
import types
from ast import literal_eval
from collections import deque
from copy import deepcopy
from functools import wraps
from random import choice
from string import ascii_letters, digits
from time import time

import attr
import idx2numpy as idx2np
import numpy as np
from click.types import ParamType
from toml import loads as _parse

from utensor_cgen.logger import logger

__all__ = ["save_idx", "save_consts", "save_graph", "log_graph",
           "NamescopedKWArgsParser", "NArgsParam", "MUST_OVERWRITE"]

class LazyLoader(types.ModuleType):

  def __init__(self, module_name='utensor_cgen', submod_name=None):
    self._module_name = '{}{}'.format(
      module_name,
      submod_name and '.{}'.format(submod_name) or ''
    )
    self._mod = None
    super(LazyLoader, self).__init__(self._module_name)

  def _load(self):
    if self._mod is None:
      self._mod = importlib.import_module(
        self._module_name
      )
    return self._mod

  def __getattr__(self, attrb):
    return getattr(self._load(), attrb)

  def __dir__(self):
    return dir(self._load())

tf = LazyLoader('tensorflow')
tf_python = LazyLoader('tensorflow', 'python.framework')

class LazyAttrib(object):

  def __init__(self, obj, attr_name):
    self._obj = obj
    self._attr_name = attr_name

  def __getattr__(self, name):
    return getattr(self.attrib, name)
  
  def __call__(self, *args, **kwargs):
    return self.attrib(*args, **kwargs)

  @property
  def attrib(self):
    return getattr(self._obj, self._attr_name)


def log_graph(graph_or_graph_def, logdir):
  from tensorflow.compat.v1 import GraphDef

  if isinstance(graph_or_graph_def, GraphDef):
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
    graph_def = tf_python.graph_util.remove_training_nodes(sess.graph_def)
    sub_graph_def = tf_python.graph_util.convert_variables_to_constants(sess=sess,
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
  def __init__(self, namespace, kwargs):
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


class _MustOverwrite(object):
  _obj = None

  def __new__(cls, *args, **kwargs):
    if cls._obj is None:
      cls._obj = object.__new__(cls, *args, **kwargs)
    return cls._obj


MUST_OVERWRITE = _MustOverwrite()


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

def prune_graph(ugraph, output_nodes=None):
  """
  Remove nodes that is no longer needed *in-place*

  this function will trace the output nodes of the
  given graph by `BFS <https://en.wikipedia.org/wiki/Breadth-first_search>`_
  and remove all nodes which are not reachable afterward

  :param ugraph: the graph to be pruned
  :type ugraph: :class:`.uTensorGraph`
  :param output_nodes: the output nodes
  :type output_nodes: List[String]
  """
  new_ugraph = deepcopy(ugraph)
  if output_nodes is None:
    output_nodes = ugraph.output_nodes[:]
  else:
    new_ugraph.output_nodes = output_nodes[:]
  # BFS to find all ops you need
  ops_in_need = set(output_nodes)
  queue = [name for name in output_nodes]
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
  topologic_order_graph(new_ugraph)
  return new_ugraph

def random_str(length=8):
  letters = ascii_letters+digits
  chars = [choice(letters) for _ in range(length)]
  return ''.join(chars)

def parse_toml(file_or_path):
  if isinstance(file_or_path, str):
    fid = open(file_or_path, 'r')
  doc = _parse(fid.read())
  fid.close()
  return doc

def timed(func):

  @wraps(func)
  def wrapped(*args, **kwargs):
    start_time = time()
    ret = func(*args, **kwargs)
    end_time = time()
    logger.info('collapsed time of calling %s: %0.4f seconds', func.__name__, end_time - start_time)
    return ret
  
  return wrapped

def is_abstract(func):
  if isinstance(func, types.MethodType):
    func = func.__func__
  return getattr(func, '__isabstractmethod__', False)


class class_property(object):
    
  def __init__(self, getter):
    self._getter = getter
      
  def __get__(self, obj, objtype=None):
    if objtype is None:
      return self._getter(obj)
    return self._getter(objtype)


@attr.s
class Pipeline(object):
  _funcs = attr.ib(factory=list)

  def __call__(self, *args, **kwargs):
    result = None
    for func in self._funcs:
      if result is None:
        result = func(*args, **kwargs)
      else:
        result = func(*result)
    return result

  def __getitem__(self, slice_obj):
    cls = type(self)
    return cls(funcs=self._funcs[slice_obj])


class Configuration(object):
  def __init__(self, defaults=None, user_config=None):
    """
    Note
    ----
    - any value that is in user_config should be in defaults
    - any value that is not in defaults should not be in user_config 
    """
    # TODO: write a check on the inputs?
    if defaults is None:
      defaults = {}
    if user_config is None:
      user_config = {}
    self._defaults = defaults
    self._user_config = user_config

  @property
  def defaults(self):
    return self._defaults
  
  @property
  def user_config(self):
    return self._user_config
  
  def get(self, key, default=None):
    value = default
    if key in self._user_config:
      value = self._user_config[key]
    elif key in self._defaults:
      value = self._defaults[key]
    return value

  def keys(self):
    return self.to_dict().keys()
  
  def values(self):
    return self.to_dict().values()

  def items(self):
    config = self.to_dict()
    return config.items()
  
  def to_dict(self):
    config = deepcopy(self._defaults)
    config.update(self._user_config)
    return config

  def __getitem__(self, key):
    if key not in self:
      raise KeyError('invalid key: {}'.format(key))
    value = self._user_config.get(
      key, self._defaults[key]
    )       
    if isinstance(value, dict):
      value = type(self)(self._defaults[key], value)
    return value

  def __contains__(self, key):
    return key in self._user_config or key in self._defaults

  def __repr__(self):
    return (
      'Configuration(\n'
      '  defaults={},\n'
      '  user_config={} \n'
      ')'
    ).format(self._defaults, self._user_config)


class must_return_type(object):

  def __init__(self, type_):
    self._expected_type = type_
  
  def __call__(self, func):
    @wraps(func)
    def wrapped(*args, **kwargs):
      ret = func(*args, **kwargs)
      ret_cls = type(ret)
      if not issubclass(ret_cls, self._expected_type):
        raise TypeError(
          "expecting {} to return value of type {}, get {}".format(
            func,
            self._expected_type,
            ret_cls
          )
        )
      return ret
    wrapped._has_return_type_check = True
    wrapped._expecting = self._expected_type
    return wrapped
  
  @staticmethod
  def get_expect_type(wrapped):
    if isinstance(wrapped, classmethod):
      wrapped = wrapped.__func__
    return wrapped._expecting

  @staticmethod
  def return_type_is_ensured(wrapped):
    if isinstance(wrapped, classmethod):
      wrapped = wrapped.__func__
    return getattr(wrapped, '_has_return_type_check', False)
