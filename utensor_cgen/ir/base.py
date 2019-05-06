# -*- coding: utf8 -*-
import re
from copy import deepcopy

import attr
import numpy as np
import six
from attr.validators import instance_of

import tensorflow as tf
from tensorflow.core.framework.attr_value_pb2 import AttrValue as _AttrValue
from tensorflow.core.framework.attr_value_pb2 import \
    NameAttrList as _NameAttrList
from tensorflow.core.framework.tensor_pb2 import TensorProto as _TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import \
    TensorShapeProto as _TensorShapeProto
from tensorflow.core.framework.types_pb2 import DataType as _DataType
from utensor_cgen.utils import topologic_order_graph

from .converter import AttrValueConverter, ConverterFactory

__all__ = [
  'TensorInfo', 'OperationInfo',
  'MetaOperationInfo', 'uTensorGraph',
  'uTensorGraphView'
]


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
          
  _ugraph = attr.ib(repr=False)
  @_ugraph.validator
  def check(self, attrib, value):
    if not isinstance(value, uTensorGraph):
      raise ValueError('Expecting a uTensorGraph, get {}'.format(type(value)))  
  
  @property
  def ugraph(self):
    return self._ugraph

  @property
  def op(self):
    return self._ugraph.ops_info.get(self.op_name, None)

  @property
  def backend(self):
    return self._ugraph.backend

  @property
  def is_dangling(self):
    op = self.op
    if not op:
      return True
    return op.is_dangling
  
  @property
  def n_th_output(self):
    if self.is_dangling:
      raise ValueError(
        "dangling tensor: {}".format(self.name)
      )
    op = self.op
    out_tnames = [t_info.name for t_info in op.output_tensors]
    return out_tnames.index(self.name)

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
  _backend = attr.ib(type=str)
  # FIXME: it's better to make OperationInfo to be instantiate without ugraph
  _ugraph = attr.ib(repr=False)
  @_ugraph.validator
  def check(self, attrib, value):
    if not isinstance(value, uTensorGraph):
      raise ValueError(
        'Expecting a uTensorGraph, '
        'get {}'.format(type(value))
      )

  input_tensors = attr.ib(validator=instance_of(list))
  @input_tensors.validator
  def check(self, attribute, value):
    # FIXME: we need a refactor of IR, allowing None here is just temporary
    # especially for graph rewrite
    if not all([isinstance(v, (TensorInfo, type(None))) for v in value]):
      raise ValueError('Expecting a list of TensorInfo for input_tensors')
  
  n_inputs = attr.ib(validator=instance_of(int))

  output_tensors = attr.ib(validator=instance_of(list))
  @output_tensors.validator
  def check(self, attribute, value):
    if not all([isinstance(v, TensorInfo) for v in value]):
      raise ValueError('Expecting a list of TensorInfo for output_tensors')

  n_outputs = attr.ib(validator=instance_of(int))

  op_type = attr.ib(type=str)
  op_attr = attr.ib(factory=dict, converter=dict)

  @property
  def ugraph(self):
    return self._ugraph
  
  @property
  def backend(self):
    return self._backend

  @property
  def input_nodes(self):
    in_ops = []
    for tensor in self.input_tensors:
      if tensor.op_name not in in_ops:
        in_ops.append(tensor.op_name)
    return [self._ugraph.ops_info.get(name, None) for name in in_ops]

  @property
  def output_nodes(self):
    out_ops = []
    for op in self._ugraph.ops:
      for in_tensor in op.input_tensors:
        if in_tensor.op_name == self.name and op.name not in out_ops:
          out_ops.append(op.name)
          break
    return [self._ugraph.ops_info[name] for name in out_ops]
  
  @property
  def is_dangling(self):
    """
    True: the op is dangling in the graph
    False: otherwise
    """
    return None in self.input_nodes

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
    self._ugraph.ops_info[self.name] = self
    if not self.n_inputs == len(self.input_tensors):
      raise ValueError('n_inputs is not equal to the length of input_tensors')
    if not self.n_outputs == len(self.output_tensors):
      raise ValueError('n_outputs is not equal to the length of output_tensors')

  def __deepcopy__(self, memo):
    op_info = OperationInfo(name=self.name,
                            input_tensors=deepcopy(self.input_tensors, memo),
                            n_inputs=self.n_inputs,
                            output_tensors=deepcopy(self.output_tensors, memo),
                            n_outputs=self.n_outputs,
                            op_type=self.op_type,
                            backend=self.backend,
                            op_attr=deepcopy(self.op_attr, memo),
                            ugraph=memo['ugraph'])
    return op_info

  def copy_into_graph(self, ugraph):
    return deepcopy(self, {'ugraph': ugraph})


class MetaOperationInfo(OperationInfo):

  def __init__(self, op_info, morphism):
    self._op_info = op_info
    self.morphism = morphism

  def __getattr__(self, name):
    return getattr(self._op_info, name)


@attr.s
class uTensorGraph(IRBase, _NoShallowCopyMixin):
  """
  Attributes
  ==========
  ops_info : dict
  topo_order : list
  output_nodes : list
  backend : str {"tensorflow", 'pytorch'(future work)}

  How to Build a uTensorGraph
  ===========================
  1. create a empty graph
    - give a list of names of output nodes (required)
    - (optional) give backend string
    - leave ops_info empty
  2. setup the ops_info
    - when you set the value of ops_info, which is an OperationInfo instance,
      make sure its ugraph attribute is the ugraph you just created at step 1
  3. pass the ugraph to topologic_order_graph to setup the order
     of the ops
  """
  KWPARSER_PATTERN = re.compile(r'^([^\d\W][\w\d_]*)__([^\d\W][\w\d_]*)')

  output_nodes = attr.ib(type=list)
  _backend = attr.ib(default='', type=str)
  ops_info = attr.ib(factory=dict)
  # non-init
  topo_order = attr.ib(factory=list, init=False)
  _type_to_op_map = attr.ib(factory=dict, init=False, repr=False)

  def __attrs_post_init__(self):
    if not self.output_nodes:
      raise ValueError('No output_nodes given')
  
  def get_ops_by_type(self, given_op_type):
    if not self._type_to_op_map:
      for op_info in self.ops_info.values():
        op_type = op_info.op_type
        ops = self._type_to_op_map.get(
          op_type,
          []
        ) + [op_info]
        self._type_to_op_map.update(
          [(op_type, ops),]
        )
    return self._type_to_op_map.get(given_op_type, [])
  
  @property
  def output_ops(self):
    return [self.ops_info[name] for name in self.output_nodes]
  
  @property
  def output_tensors(self):
    out_tensors = []
    for op in self.output_ops:
      for tensor in op.output_tensors:
        out_tensors.append(tensor)
    return out_tensors

  @property
  def input_ops(self):
    ops = []
    for op in self.ops_info.values():
      if not op.input_tensors:
        ops.append(op)
    return ops
  
  @property
  def input_tensors(self):
    in_tensors = []
    for op in self.input_ops:
      for tensor in op.input_tensors:
        in_tensors.append(tensor)
    return in_tensors
  
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
    if not self.topo_order:
      topologic_order_graph(self)
    return [self.ops_info[name] for name in self.topo_order]

  def add_op(self, op):
    if not isinstance(op, OperationInfo):
      raise ValueError('expecting OperationInfo, get {}'.format(type(op)))
    if op.name in self.ops_info:
      raise ValueError('duplicate op detected, {}'.format(op.name))
    op.ugraph = self

    # if(op.name == 'convert_uint8_q7_Relu/eightbit_transpose_0_q7'):
    #   import pdb; pdb.set_trace()
    self.ops_info[op.name] = op
    topologic_order_graph(self)

  def drop_op(self, op_name):
    if op_name not in self.ops_info:
      raise ValueError('op not found in the graph: {}'.format(op_name))
    del self.ops_info[op_name]
    self.topo_order.remove(op_name)

  def __deepcopy__(self, memo):
    new_graph = uTensorGraph(output_nodes=self.output_nodes)
    memo['ugraph'] = new_graph

    new_graph.ops_info = {
      k: deepcopy(v, memo)
      for k, v in self.ops_info.items()
    }
    new_graph._backend = self._backend
    topologic_order_graph(new_graph)
    return new_graph


@attr.s
class uTensorGraphView(IRBase, _NoShallowCopyMixin):

  _ugraph = attr.ib(type=uTensorGraph)
  _op_names = attr.ib(type=list)
  output_nodes = attr.ib(type=list)

  topo_order = attr.ib(init=False, factory=list)
  ops_info = attr.ib(init=False, factory=dict)

  def __attrs_post_init__(self):
    for name in self._op_names:
      self.ops_info[name] = self._ugraph.ops_info[name]
    topologic_order_graph(self)
  
  @property
  def backend(self):
    return self._ugraph.backend

  @property
  def input_ops(self):
    ops = []
    for name in self.topo_order:
      op = self.ops_info[name]
      input_tensors = op.input_tensors
      if all([
        tensor.op.name not in self.ops_info
        for tensor in input_tensors
      ]):
        ops.append(op)
    return ops
  
  @property
  def input_tensors(self):
    in_tensors = []
    for op in self.input_ops:
      for tensor in op.input_tensors:
        in_tensors.append(tensor)
    return in_tensors
  
  @property
  def output_ops(self):
    return [self.ops_info[name] for name in self.output_nodes]
  
  @property
  def output_tensors(self):
    out_tensors = []
    for op in self.output_ops:
      for tensor in op.output_tensors:
        out_tensors.append(tensor)
    return out_tensors
