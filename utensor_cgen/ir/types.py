# -*- coding: utf8 -*-
import six
from copy import deepcopy

import attr
from attr import validators
import numpy as np
import tensorflow as tf
from tensorflow.core.framework.tensor_pb2 import TensorProto as _TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto as _TensorShapeProto
from tensorflow.core.framework.attr_value_pb2 import (AttrValue as _AttrValue,
                                                      NameAttrList as _NameAttrList)
from tensorflow.core.framework.types_pb2 import DataType as _DataType
from tensorflow.contrib.util import make_ndarray
from tensorflow.tools.graph_transforms import TransformGraph

from .converter import ConverterFactory
from .utils import parse_tensor_name


__all__ = ['TensorInfo', 'OperationInfo', 'uTensorGraph']

class _NoShallowCopyMixin(object):

  def __copy__(self):
    raise NotImplementedError('shallow copy is not allowed for type %s' % type(self))


class IRBase(object):
  # shared helper functions
  def _is_py_generic_type(self, value):
    return type(value).__module__ == '__builtin__'
  
  def _is_protobuf_obj(self, value):
    return (hasattr(value, 'CopyFrom') or
            type(value).__module__.startswith('google.protobuf'))


@attr.s
class TensorInfo(IRBase, _NoShallowCopyMixin):
  name = attr.ib(validator=validators.instance_of(six.text_type))
  dtype = attr.ib(validator=validators.instance_of(tf.DType))

  shape = attr.ib()
  @shape.validator
  def check(self, attrib, value):
    if value is not None and not isinstance(value, list):
      raise ValueError('shape must be None or list')

  backend = attr.ib()
  @backend.validator
  def check(self, attrib, value):
    if value not in ['tensorflow']:
      raise ValueError('Unsupport backend: {}'.format(value))

  tensor_attr = attr.ib(default=None)
  
  def __attrs_post_init__(self):
    if not self.tensor_attr:
      self.tensor_attr = {}
      return

    tensor_attr = {}
    if self._is_protobuf_obj(self.tensor_attr):
      for key, attr in self.tensor_attr.items():
        if self._is_protobuf_obj(attr):
          cvt = ConverterFactory(attr)
          attr = cvt.get_generic_value()
        tensor_attr[key] = attr
    self.tensor_attr = tensor_attr

  def __deepcopy__(self, memo):
    new_tensor_attr = {}
    for k, value in self.tensor_attr.items():
      new_value = deepcopy(value, memo)
      new_tensor_attr[k] = new_value
    return TensorInfo(name=self.name,
                      dtype=self.dtype,
                      shape=self.shape,
                      backend=self.backend,
                      tensor_attr=new_tensor_attr)

  # legacy code: to make it works like namedtuple
  def __iter__(self):
    # TODO remove all such code in utensor
    #   name, dtype, shape = tensor_info
    return iter((self.name, self.dtype, self.shape))

  def __getitem__(self, k):
    return (self.name, self.dtype, self.shape)[k]


@attr.s
class OperationInfo(IRBase, _NoShallowCopyMixin):

  name = attr.ib(validator=validators.instance_of(six.text_type))

  input_tensors = attr.ib(validator=validators.instance_of(list))
  @input_tensors.validator
  def check(self, attribute, value):
    if not all([isinstance(v, TensorInfo) for v in value]):
      raise ValueError('Expecting a list of TensorInfo for input_tensor')

  output_tensors = attr.ib()
  @output_tensors.validator
  def check(self, attribute, value):
    assert isinstance(value, list), \
      "output_tensor should be of type %s, get %s" % (list, type(value))
    if not all([isinstance(v, TensorInfo) for v in value]):
      raise ValueError('Expecting a list of TensorInfo for output_tensor')

  op_type = attr.ib(validator=validators.instance_of(six.text_type))

  backend = attr.ib()
  @backend.validator
  def check(self, attribute, value):
    if value not in ['tensorflow']:
      raise ValueError('Unsupported backend: {}'.format(value))

  # misc dict
  op_attr = attr.ib(default=None)

  def __attrs_post_init__(self):
    if not self.op_attr:
      self.op_attr = {}
      return

    if self._is_protobuf_obj(self.op_attr):
      op_attr = {}
      for key, attr in self.op_attr.items():
        if self._is_protobuf_obj(attr):
          cvt = ConverterFactory(attr)
          attr = cvt.get_generic_value()
        op_attr[key] = attr
      self.op_attr = op_attr
  
  def __deepcopy__(self, memo):
    return OperationInfo(name=self.name,
                         input_tensors=deepcopy(self.input_tensors, memo),
                         output_tensors=deepcopy(self.output_tensors, memo),
                         op_type=self.op_type,
                         backend=self.backend,
                         op_attr=deepcopy(self.op_attr, memo))


class uTensorGraph(IRBase, _NoShallowCopyMixin):
  """
  Attributes
  ==========
  ops_info : dict
  topo_order : list
  output_nodes : list
  _backend : str
  """
  def __init__(self, graph=None, output_nodes=None):
    assert isinstance(output_nodes, list) or output_nodes is None, \
        "output_nodes should be of type %s or None, get %s" % (list, type(output_nodes))
    if graph is None:
      # empty graph
      self.ops_info = {}
      self.topo_order = []
      self.output_nodes = []
    elif isinstance(graph, tf.GraphDef):
      if not output_nodes:
        raise ValueError('No output_nodes given')
      self._init_from_graph_def(graph, output_nodes)
      self.output_nodes = output_nodes
    else:
      raise ValueError('Only support tensorflow now')

  @property
  def graph_def(self):
    assert self._backend == 'tensorflow', \
      'Convert a uTensorGraph to tf.GraphDef from a non-tf backend'
    graph_def = tf.GraphDef()
    for node_name in self.topo_order:
      op_info = self.ops_info[node_name]
      attr = {}
      for key, obj in op_info.op_attr.items():
        if key.startswith('_'):
          continue
        value_name = obj.value_name
        value = obj.value
        attr_value = _AttrValue(**{value_name: value})
        attr[key] = attr_value
      graph_def.node.add(name=op_info.node_name,
                         op=op_info.op_type,
                         input=[in_tensor.name for in_tensor in op_info.input_tensors],
                         device=op_info.op_attr.get('_device', ''),
                         attr=attr)
    return graph_def
  
  @property
  def ops(self):
    return [self.ops_info[name] for name in self.topo_order]
  
  @staticmethod
  def _topologic_order_graph(ugraph):
    # https://en.wikipedia.org/wiki/Topological_sorting
    queue = deepcopy(ugraph.output_nodes)
    visited = set()    # temporary mark
    perm_visit = set()  # Permanent mark
    ops_torder = []  # L

    def visit(node_name):
      if node_name in perm_visit:
        return
      if node_name in visited:
        raise ValueError("Input graph is not a DAG")

      visited.add(node_name)
      op_info = ugraph.ops_info[node_name]

      for t_info in op_info.input_tensors:
        op_name = parse_tensor_name(t_info.name)[0]
        visit(op_name)

      perm_visit.add(node_name)
      ops_torder.insert(0, node_name)

    while queue:
      node_name = queue.pop(0)
      visit(node_name)
    return ops_torder

  @staticmethod
  def _parse_tf_tshape(t_shape):
    try:
      shape = t_shape.as_list()
    except ValueError:
      shape = None
    return shape

  # tensorflow
  def _init_from_graph_def(self, graph_def, output_nodes):
    """Tensorflow
    """
    if not self._tf_is_freeze_graph(graph_def):
      raise ValueError('Given graph_def is not freezed')
    self._backend = 'tensorflow'
    self.ops_info = {}
    self.topo_order = []
    graph = tf.Graph()
    with graph.as_default():
      tf.import_graph_def(graph_def, name='')
    graph_def = TransformGraph(graph_def,
                               [],
                               output_nodes,
                               ['sort_by_execution_order'])
    for node in graph_def.node:
      op = graph.get_operation_by_name(node.name)
      in_tensors = [TensorInfo(name=tensor.name,
                               dtype=tensor.dtype,
                               shape=self._parse_tf_tshape(tensor.shape),
                               backend='tensorflow')
                    for tensor in op.inputs]
      out_tensors = [TensorInfo(name=tensor.name,
                                dtype=tensor.dtype,
                                shape=self._parse_tf_tshape(tensor.shape),
                                backend='tensorflow')
                     for tensor in op.outputs]
      op_type = node.op
      op_attr = node.attr
      op_info = OperationInfo(name=node.name,
                              input_tensors=in_tensors,
                              output_tensors=out_tensors,
                              op_type=op_type,
                              backend='tensorflow',
                              op_attr=op_attr)
      op_info.op_attr['_device'] = node.device
      self.ops_info[node.name] = op_info
      self.topo_order.append(node.name)
  
  def _tf_is_freeze_graph(self, graph_def):
    is_frozen = all(node.op not in ['VariableV2'] for node in graph_def.node)
    return is_frozen

  def __deepcopy__(self, memo):
    new_graph = uTensorGraph()
    new_ops_info = dict((k, deepcopy(v, memo)) for k, v in self.ops_info.items())
    new_topo_order = [name for name in self.topo_order]

    new_graph.ops_info = new_ops_info
    new_graph.topo_order = new_topo_order
    new_graph.output_nodes = self.output_nodes
    new_graph._backend = self._backend
    return new_graph
