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


__all__ = ['TensorInfo', 'OperationInfo', 'uTensorGraph']

class _NoShallowCopy(object):
  def __copy__(self):
    raise NotImplementedError('shallow copy is not allowed for type %s' % type(self))

@attr.s
class TensorInfo(_NoShallowCopy):
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

  tensor_attr = attr.ib(default=attr.Factory(dict),
                        validator=validators.instance_of(dict))

  def __deepcopy__(self):
    new_tensor_attr = {}
    for k, value in self.tensor_attr.items():
      if hasattr(value, 'CopyFrom'):
        # protobuf object
        new_value = type(value)()
        new_value.CopyFrom(value)
      else:
        new_value = deepcopy(value)
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
class OperationInfo(_NoShallowCopy):
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
    if self.op_attr:
      if self.backend == 'tensorflow':
        # parse tensorflow NodeDef.attr (a protobuf map)
        self.op_attr = self._tf_attr_to_dict(self.op_attr)
      else:
        raise ValueError('Only support tensorflow now, FIX THIS!')

  def _tf_attr_to_dict(self, op_attr):
    ret_d = {}
    for key, attr in op_attr.items():
      value_name = attr.WhichOneof('value')
      value = getattr(attr, value_name)
      value = self._tf_convert_to_py_generic(value)
      # FIXME: a dict is not a very consice data structure
      # Maybe we should come up with some other kind of object
      # when we add PyTorch support in uTensor cli
      ret_d[key] = {'value_name': value_name,
                    'value': value}
    return ret_d

  def _tf_convert_to_py_generic(self, value):
    if self._is_py_generic_type(value):
      return value
    assert hasattr(value, 'CopyFrom'), \
      'Expecting a protobuf object, get %s: %s' % (type(value), value)
    # TODO: convert protobuf object to generic python representation
    # generic representation := using only python builtin types or types
    #                           defined in utensor_cgen
    return value
  
  def _is_py_generic_type(self, value):
    return type(value).__module__ == '__builtin__'


class uTensorGraph(object):
  """
  Attributes
  ==========
  ops_info : dict
  topo_order : list
  output_nodes : list
  _backend : str
  """
  def __init__(self, graph=None, output_nodes=None):
    if graph is None:
      # empty graph
      self.ops_info = {}
      self.topo_order = []
      self.output_nodes = []
    elif isinstance(graph, tf.GraphDef):
      assert isinstance(output_nodes, list), \
        "output_nodes should be of type %s, get %s" % (list, type(output_nodes))
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
        value_name = obj['value_name']
        value = obj['value']
        attr_value = _AttrValue(**{value_name: value})
        attr[key] = attr_value
      tf_node = graph_def.node.add(name=op_info.node_name,
                                   op=op_info.op_type,
                                   input=[in_tensor.name for in_tensor in op_info.input_tensors],
                                   device=op_info.op_attr.get('_device', ''),
                                   attr=attr)
    return graph_def
  
  @property
  def ops(self):
    return [self.ops_info[name] for name in self.topo_order]

  @staticmethod
  def _parse_tf_tshape(t_shape):
    try:
      shape = t_shape.as_list()
    except ValueError:
      shape = None
    return shape
  
  def __deepcopy__(self):
    new_graph = uTensorGraph()
    new_ops_info = dict((k, deepcopy(v)) for k, v in self.ops_info.items())
    new_topo_order = [name for name in self.topo_order]

    new_graph.ops_info = new_ops_info
    new_graph.topo_order = new_topo_order
    new_graph.output_nodes = self.output_nodes
    new_graph._backend = self._backend
    return new_graph

  # tensorflow
  def _init_from_graph_def(self, graph_def, output_nodes):
    """Tensorflow
    """
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
