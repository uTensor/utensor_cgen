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
from utensor_cgen.utils import random_str, topologic_order_graph

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


@attr.s(cmp=False)
class TensorInfo(IRBase, _NoShallowCopyMixin):
  """
  name : str
  dtype : numpy.dtype
  shape : list

  TODO: the need for null tensor info, that is,
  a tensor which may not be attached to an op
  """
  name = attr.ib(validator=instance_of(six.string_types))
  op_name = attr.ib(validator=instance_of(six.string_types))
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
  
  _NULL_PREFIX = 'utensor_null'

  @classmethod
  def make_null_tensor(cls, ugraph):
    op_name = '{}_{}'.format(cls._NULL_PREFIX, random_str())
    name = '{}:0'.format(op_name)
    return cls(
      name=name,
      op_name=op_name,
      dtype=np.dtype('float'),
      shape=None,
      ugraph=ugraph
    )
  
  @property
  def is_null_tensor(self):
    return self.op_name.startswith(self._NULL_PREFIX)

  def __deepcopy__(self, memo):
    new_tensor = TensorInfo(name=self.name,
                            ugraph=memo['ugraph'],
                            op_name=self.op_name,
                            dtype=self.dtype,
                            shape=deepcopy(self.shape, memo))
    return new_tensor
  
  def __hash__(self):
    return hash(self.name)
  
  def __eq__(self, other):
    if not isinstance(other, type(self)):
      return False
    return (self.name == other.name) and (self._ugraph is other._ugraph)

  def move_into(self, ugraph):
    """
    Move Synmatic of the OperationInfo objects
    """
    self._ugraph = ugraph

@attr.s(cmp=False)
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
      if tensor.op is None:
        continue
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

  def add_null_input_tensor(self, idx=-1):
    if self.op_type != 'Placeholder':
      raise ValueError(
        'can only add null tensor to op of type Placeholder: %s' % self.op_type
      )
    if idx > len(self.input_tensors):
      raise ValueError(
        "can't insert null tensor at {} as {} input tensors present".format(
          idx, len(self,input_tensors)
        )
      )
    null_tensor = TensorInfo.make_null_tensor(ugraph=self._ugraph)
    self.input_tensors.insert(idx, null_tensor)
    self.n_inputs += 1
    return null_tensor
  
  def replace_with_null_input_tensor(self, idx):
    if idx >= len(self.input_tensors):
      raise ValueError(
        'index out of bound: %s' % idx
      )
    self.input_tensors[idx] = TensorInfo.make_null_tensor(ugraph=self._ugraph)

  def move_into(self, ugraph):
    """
    Move Synmatic of the OperationInfo objects
    """
    self._ugraph = ugraph
    for tensor in self.input_tensors:
      tensor.move_into(ugraph)
    for tensor in self.output_tensors:
      tensor.move_into(ugraph)
  
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

  def __hash__(self):
    return hash(self.name)
  
  def __eq__(self, other):
    if not isinstance(other, type(self)):
      return False
    return (self.name == other.name) and (self._ugraph is other._ugraph)
  
  def __getitem__(self, tensor_idx):
    num_out_tensors = len(self.output_tensors)
    if tensor_idx > num_out_tensors:
      raise IndexError(
        'index out of bound: {} out of {}'.format(tensor_idx, num_out_tensors)
      )
    return self.output_tensors[tensor_idx]


class MetaOperationInfo(OperationInfo):

  def __init__(self, op_info, morphism):
    self._op_info = op_info
    self.morphism = morphism

  def __getattr__(self, name):
    return getattr(self._op_info, name)


@attr.s(cmp=False)
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
    out_tensors = set([])
    for op in self.output_ops:
      for tensor in op.output_tensors:
        out_tensors.add(tensor)
    return out_tensors

  @property
  def input_ops(self):
    ops = []
    for op in self.ops_info.values():
      if (
        not op.input_tensors 
        or any([tensor.is_null_tensor for tensor in op.input_tensors])
      ):
        ops.append(op)
    return ops
  
  @property
  def input_tensors(self):
    in_tensors = set([])
    for op in self.input_ops:
      for tensor in op.input_tensors:
        in_tensors.add(tensor)
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

  def add_op(self, op, sort=True):
    if not isinstance(op, OperationInfo):
      raise ValueError('expecting OperationInfo, get {}'.format(type(op)))
    if op.name in self.ops_info:
      raise ValueError('duplicate op detected, {}'.format(op.name))
    op._ugraph = self

    self.ops_info[op.name] = op

    # FIXME: forcing a topo-order here prevent us from dynamic-graph-construction
    # The temporary fix is to disable this as an option
    if sort:
      topologic_order_graph(self)

  def drop_op(self, op_name):
    """Experimental, don't use.
    """
    if op_name not in self.ops_info:
      raise ValueError('op not found in the graph: {}'.format(op_name))
    del self.ops_info[op_name]
    self.topo_order.remove(op_name)
  
  def merge_into(self, other_ugraph):
    """
    NOTE:(IMPORTANT) after this method call, you should
    consider both the graph and the other ugraph is at a
    dangling state, which means the output nodes may not
    be correct, the topological ordering may not be correct
    ...etc. After all modifications to the graph are done,
    you may need to call following functions in order to
    maintain the state of the graph:
    
    1. prune_graph (from `utensor_cgen.utils`)
    2. topologic_order_graph (from `utensor_cgen.utils`)
    """
    for op in self.ops_info.values():
      op._ugraph = other_ugraph
      for input_tensor in op.input_tensors:
        input_tensor.move_into(other_ugraph)
      for output_tensor in op.output_tensors:
        output_tensor.move_into(other_ugraph)
      if op.op_type not in self._type_to_op_map:
        self._type_to_op_map[op.op_type] = []
      self._type_to_op_map[op.op_type].append(op)

  def __deepcopy__(self, memo):
    new_graph = uTensorGraph(
      output_nodes=self.output_nodes,
      backend=self._backend
    )
    memo['ugraph'] = new_graph

    new_graph.ops_info = {
      k: deepcopy(v, memo)
      for k, v in self.ops_info.items()
    }
    topologic_order_graph(new_graph)
    return new_graph

  def __getitem__(self, op_name):
    if op_name not in self.ops_info:
      raise KeyError('{} not found in the graph'.format(op_name))
    return self.ops_info[op_name]


@attr.s(cmp=False)
class uTensorGraphView(IRBase, _NoShallowCopyMixin):

  _ugraph = attr.ib(type=uTensorGraph)
  _op_names = attr.ib(type=list)
  output_nodes = attr.ib(type=list)
  ops_info = attr.ib(init=False, factory=dict)

  def __attrs_post_init__(self):
    for name in self._op_names:
      self.ops_info[name] = self._ugraph.ops_info[name]
  
  @property
  def backend(self):
    return self._ugraph.backend

  @property
  def input_ops(self):
    ops = set([])
    for name in self.ops_info:
      op = self.ops_info[name]
      input_tensors = op.input_tensors
      if all([
        tensor.op.name not in self.ops_info
        for tensor in input_tensors
      ]):
        ops.add(op)
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

  def __getitem__(self, op_name):
    if op_name not in self.ops_info:
      raise KeyError('{} not found in the graph view'.format(op_name))
    return self.ops_info[op_name]
