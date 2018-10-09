# -*- coding: utf8 -*-
import re
from collections import defaultdict
from copy import deepcopy

import attr
import numpy as np
import six
import tensorflow as tf
from attr.validators import instance_of
from tensorflow.contrib.util import make_ndarray
from tensorflow.core.framework.attr_value_pb2 import AttrValue as _AttrValue
from tensorflow.core.framework.attr_value_pb2 import \
    NameAttrList as _NameAttrList
from tensorflow.core.framework.tensor_pb2 import TensorProto as _TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import \
    TensorShapeProto as _TensorShapeProto
from tensorflow.core.framework.types_pb2 import DataType as _DataType
from tensorflow.tools.graph_transforms import TransformGraph

from utensor_cgen.utils import parse_tensor_name

from .converter import AttrValueConverter, ConverterFactory

__all__ = ['TensorInfo', 'OperationInfo', 'uTensorGraph']


class _NoShallowCopyMixin(object):

  def __copy__(self):
    raise RuntimeError('shallow copy is not allowed for type %s' % type(self))


class IRBase(object):

  @property
  def all_supported_backends(self):
    return ['tensorflow']


@attr.s
class TensorInfo(IRBase, _NoShallowCopyMixin):
  """
  name : str
  dtype : numpy.dtype
  shape : list
  """
  name = attr.ib(validator=instance_of(six.text_type))
  op_name = attr.ib(validator=instance_of(six.text_type))
  dtype = attr.ib(validator=instance_of(np.dtype))
  shape = attr.ib(validator=instance_of((list, type(None))))
  @shape.validator
  def check(self, attrib, shape_values):
    if shape_values is not None:
      for v in shape_values:
        assert isinstance(v, (int, type(None))), \
          "shape should be a list of integers"
  ugraph = attr.ib()
  @ugraph.validator
  def check(self, attrib, value):
    if not isinstance(value, uTensorGraph):
      raise ValueError('Expecting a uTensorGraph, get {}'.format(type(value)))
  
  @property
  def op(self):
    return self.ugraph.ops_info.get(self.op_name, None)

  @property
  def is_dangling(self):
    op = self.op
    if not op:
      return True
    return op.is_dangling

  def __deepcopy__(self, memo):
    new_tensor = TensorInfo(name=self.name,
                            ugraph=memo['ugraph'],
                            op_name=self.op_name,
                            dtype=self.dtype,
                            shape=deepcopy(self.shape, memo))
    return new_tensor


@attr.s
class OperationInfo(IRBase, _NoShallowCopyMixin):
  """
  name : str
  input_tensors : List[TensorInfo]
  output_tensors : List[TensorInfo]
  input_nodes : Set[OperationInfo]
  output_nodes : Set[OperationInfo]
  op_type : str
  backend : str {"tensorflow", 'pytorch'(future work)}
  op_attr : dict

  Note
  ====
  - `op_attr` will be a dictionary with key as str and value as generic
    types defined in `converter.ConverterFactor.all_generic_types`. The
    only exception is the key which match regex pattern r'_[^_]*'. The 
    values of such keys will be saved as-is without any type conversion.
  """
  name = attr.ib(type=str)
  ugraph = attr.ib(repr=False)
  @ugraph.validator
  def check(self, attrib, value):
    if not isinstance(value, uTensorGraph):
      raise ValueError(('Expecting a uTensorGraph, '
                        'get {}'.format(type(value))))

  input_tensors = attr.ib(validator=instance_of(list))
  @input_tensors.validator
  def check(self, attribute, value):
    if not all([isinstance(v, TensorInfo) for v in value]):
      raise ValueError('Expecting a list of TensorInfo for input_tensor')

  output_tensors = attr.ib(validator=instance_of(list))
  @output_tensors.validator
  def check(self, attribute, value):
    if not all([isinstance(v, TensorInfo) for v in value]):
      raise ValueError('Expecting a list of TensorInfo for output_tensor')

  op_type = attr.ib(type=str)

  backend = attr.ib(type=str)
  @backend.validator
  def check(self, attribute, value):
    if value not in ['tensorflow']:
      raise ValueError('Unsupported backend: {}'.format(value))

  op_attr = attr.ib(factory=dict, converter=dict)

  @property
  def input_nodes(self):
    in_ops = []
    for tensor in self.input_tensors:
      if tensor.op_name not in in_ops:
        in_ops.append(tensor.op_name)
    return [self.ugraph.ops_info.get(name, None) for name in in_ops]
  
  @property
  def output_nodes(self):
    out_ops = []
    for op in self.ugraph.ops:
      for in_tensor in op.input_tensors:
        if in_tensor.op_name == self.name and op.name not in out_ops:
          out_ops.append(op.name)
          break
    return [self.ugraph.ops_info[name] for name in out_ops]
  
  @property
  def is_dangling(self):
    """
    True: the op is dangling in the graph
    False: otherwise
    """
    return None in self.input_nodes

  @property
  def n_inputs(self):
    return len(self.input_tensors)
  
  @property
  def n_outputs(self):
    return len(self.output_tensors)

  def __attrs_post_init__(self):
    skip_pattern = re.compile(r'_utensor_[^_]*')
    if self.op_attr:
      op_attr = {}
      for k, v in self.op_attr.items():
        match = skip_pattern.match(k)
        if match:
          op_attr[k] = v
        else:
          op_attr[k] = ConverterFactory.get_generic_value(v)
      self.op_attr = op_attr
    self.ugraph.ops_info[self.name] = self

  def __deepcopy__(self, memo):
    op_info = OperationInfo(name=self.name,
                            input_tensors=deepcopy(self.input_tensors, memo),
                            output_tensors=deepcopy(self.output_tensors, memo),
                            op_type=self.op_type,
                            backend=self.backend,
                            op_attr=deepcopy(self.op_attr, memo),
                            ugraph=memo['ugraph'])
    return op_info

  def copy_into_graph(self, ugraph):
    return deepcopy(self, {'ugraph': ugraph})

class uTensorGraph(IRBase, _NoShallowCopyMixin):
  """
  Attributes
  ==========
  ops_info : dict
  topo_order : list
  output_nodes : list
  backend : str {"tensorflow", 'pytorch'(future work)}
  """
  KWPARSER_PATTERN = re.compile(r'^([^\d\W][\w\d_]*)__([^\d\W][\w\d_]*)')

  def __init__(self, graph=None, output_nodes=None):
    if output_nodes is None:
      output_nodes = []
    if graph is None:
      self.ops_info = {}
      self.topo_order = []
      self.output_nodes = []
      self._backend = ''
      return
    assert isinstance(output_nodes, list), \
        "output_nodes should be of type %s, get %s" % (list, type(output_nodes))
    if isinstance(graph, tf.GraphDef):
      if not output_nodes:
        raise ValueError('No output_nodes given')
      self._init_from_graph_def(graph, output_nodes)
    else:
      raise ValueError('Only support tensorflow now')
  
  @property
  def backend(self):
    return self._backend

  @property
  def graph_def(self):
    assert self._backend == 'tensorflow', \
      'Convert a uTensorGraph to tf.GraphDef from a non-tf backend'
    graph_def = tf.GraphDef()
    for node_name in self.topo_order:
      op_info = self.ops_info[node_name]
      attr = {}
      for key, obj in op_info.op_attr.items():
        if self.KWPARSER_PATTERN.match(key):
          continue
        value_name = obj.value_name
        tf_value = ConverterFactory.get_tf_value(obj.value)
        attr_value = _AttrValue(**{value_name: tf_value})
        attr[key] = attr_value
      graph_def.node.add(name=op_info.name,
                         op=op_info.op_type,
                         input=[in_tensor.name for in_tensor in op_info.input_tensors],
                         device=op_info.op_attr.get('tensorflow__device', ''),
                         attr=attr)
    return graph_def
  
  @property
  def ops(self):
    return [self.ops_info[name] for name in self.topo_order]

  def viz_graph(self, fname="graph.gv"):
    from graphviz import Digraph
    dot = Digraph()
    nodes = {}
    i = 0
    for node in self.ops:
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
    for node in self.ops:
        for n in node.input_tensors:
            dot.edge(nodes[n.name], nodes[node.name])
        for n in node.output_tensors:
            dot.edge(nodes[node.name], nodes[n.name])
    dot.render(fname, view=True)


  def add_op(self, op):
    if not isinstance(op, OperationInfo):
      raise ValueError('expecting OperationInfo, get {}'.format(type(op)))
    if op.name in self.ops_info:
      raise ValueError('duplicate op detected, {}'.format(op.name))
    self.ops_info[op.name] = op
    self._topologic_order_graph()

  def drop_op(self, op_name):
    if op_name not in self.ops_info:
      raise ValueError('op not found in the graph: {}'.format(op_name))
    del self.ops_info[op_name]
    self.topo_order.remove(op_name)

  def _topologic_order_graph(self):
    # https://en.wikipedia.org/wiki/Topological_sorting
    queue = deepcopy(self.output_nodes)
    visited = set()    # temporary mark
    perm_visit = set()  # Permanent mark
    ops_torder = []  # L

    def visit(node_name):
      if node_name in perm_visit:
        return
      if node_name in visited:
        raise ValueError("Input graph is not a DAG")

      visited.add(node_name)
      op_info = self.ops_info[node_name]

      for t_info in op_info.input_tensors:
        visit(t_info.op_name)

      perm_visit.add(node_name)
      ops_torder.insert(0, node_name)

    while queue:
      node_name = queue.pop(0)
      visit(node_name)
    self.topo_order = ops_torder[::-1]

  # tensorflow
  @staticmethod
  def _tf_parse_tshape(t_shape):
    try:
      shape = t_shape.as_list()
    except ValueError:
      shape = None
    return shape

  def _init_from_graph_def(self, graph_def, output_nodes):
    """Initailize graph with Tensorflow GraphDef
    """
    if not self._tf_is_freeze_graph(graph_def):
      raise ValueError('Given graph_def is not freezed')
    self._backend = 'tensorflow'
    self.ops_info = {}
    self.topo_order = []
    self.output_nodes = output_nodes
    graph = tf.Graph()
    with graph.as_default():
      tf.import_graph_def(graph_def, name='')

    for node in graph_def.node:
      op = graph.get_operation_by_name(node.name)
      in_tensors = [TensorInfo(name=tensor.name,
                               ugraph=self,
                               op_name=tensor.op.name,
                               dtype=np.dtype(tensor.dtype.as_numpy_dtype),
                               shape=self._tf_parse_tshape(tensor.shape))
                    for tensor in op.inputs]
      out_tensors = [TensorInfo(name=tensor.name,
                                ugraph=self,
                                op_name=op.name,
                                dtype=np.dtype(tensor.dtype.as_numpy_dtype),
                                shape=self._tf_parse_tshape(tensor.shape))
                     for tensor in op.outputs]
      op_type = node.op
      op_attr = node.attr
      op_info = OperationInfo(name=node.name,
                              input_tensors=in_tensors,
                              output_tensors=out_tensors,
                              op_type=op_type,
                              backend='tensorflow',
                              op_attr=op_attr,
                              ugraph=self)
      op_info.op_attr['tensorflow__device'] = node.device
      self.ops_info[node.name] = op_info
    self._topologic_order_graph()
  
  def _tf_is_freeze_graph(self, graph_def):
    is_frozen = all(node.op not in ['VariableV2'] for node in graph_def.node)
    return is_frozen

  def __deepcopy__(self, memo):
    new_graph = uTensorGraph()
    memo['ugraph'] = new_graph
    new_ops_info = dict((k, deepcopy(v, memo)) for k, v in self.ops_info.items())
    new_topo_order = [name for name in self.topo_order]

    new_graph.ops_info = new_ops_info
    new_graph.topo_order = new_topo_order
    new_graph.output_nodes = self.output_nodes
    new_graph._backend = self._backend
    return new_graph
