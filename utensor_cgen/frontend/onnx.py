import os
import re
from collections import Counter
from numbers import Number

import numpy as np
import onnx
import tensorflow.compat.v1 as tf
from onnx import mapping, numpy_helper
from onnx.onnx_pb import TensorProto

from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.frontend.base import Parser
from utensor_cgen.ir import OperationInfo, TensorInfo, uTensorGraph
from utensor_cgen.ir.converter import AttrValueConverter, TensorProtoConverter
from utensor_cgen.legalizer import Legalizer
from utensor_cgen.utils import topologic_order_graph


def _convert_op_attribute(attrib_pb):
  # TODO: integrate with ir.converter
  if attrib_pb.HasField('f'):
    return attrib_pb.f
  elif attrib_pb.HasField('i'):
    return attrib_pb.i
  elif attrib_pb.HasField('s'):
    return attrib_pb.s
  elif attrib_pb.floats:
    return TensorProtoConverter.__utensor_generic_type__(np_array=np.array(attrib_pb.floats))
  elif attrib_pb.ints:
    return TensorProtoConverter.__utensor_generic_type__(np_array=np.array(attrib_pb.ints))
  elif attrib_pb.strings:
    return TensorProtoConverter.__utensor_generic_type__(np_array=np.array(attrib_pb.strings))
  else:
    raise ValueError('Unknown attribute value: {}'.format(attrib_pb))

# I stole these code snippets from [onnx-tf](https://github.com/onnx/onnx-tensorflow)
# I have to do so since TF2.0 fuck up onnx-tf's backend
def _onnx_dtype(dtype):
  if isinstance(dtype, Number):
    onnx_dype = dtype
  elif isinstance(dtype, str):
    onnx_dype = TensorProto.DataType.Value(dtype)
  else:
    raise RuntimeError("dtype should be number or str.")
  return onnx_dype

def onnx2tf(dtype):
  return tf.as_dtype(mapping.TENSOR_TYPE_TO_NP_TYPE[_onnx_dtype(dtype)])

def _onnx_initializer_to_input_dict_items(initializer):
  """ Convert ONNX graph initializer to input dict items.

  :param initializer: ONNX graph initializer, list of TensorProto.
  :return: List of input dict items.
  """
  def tensor2list(onnx_tensor):
    # Use the onnx.numpy_helper because the data may be raw
    return numpy_helper.to_array(onnx_tensor).flatten().tolist()

  return [(init.name,
            tf.constant(
                tensor2list(init),
                shape=init.dims,
                dtype=onnx2tf(init.data_type)))
          for init in initializer]


@FrontendSelector.register(target_exts=['.onnx'])
class OnnxParser(Parser):
  # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
  # https://pytorch.org/docs/stable/onnx.html

  def parse(self, onnx_file, output_nodes=None, model_name=None):
    tf.disable_eager_execution()
    if model_name:
      graph_name = model_name
    else:
      graph_name, _ = os.path.splitext(
        os.path.basename(onnx_file)
      )
    tf.reset_default_graph()
    model = onnx.load(onnx_file)
    onnx_graph = model.graph
    ugraph = uTensorGraph(
      name=graph_name,
      output_nodes=[],
      lib_name='onnx',
      ops_info={},
    )
    self._build_graph(onnx_graph, ugraph)
    ugraph = Legalizer.legalize(ugraph)
    tf.reset_default_graph()
    return ugraph

  def _build_graph(self, onnx_graph, ugraph):
    op_types_cnt = Counter() # op_type (str) -> count (int)
    tensor_names_map = {}    # tensor_name (str) -> tensor_info (TensorInfo)
    # these methods will update inputs **inplace**
    self._build_param_ops(
      onnx_graph, ugraph, op_types_cnt, tensor_names_map
    )
    self._build_input_ops(
      onnx_graph, ugraph, op_types_cnt, tensor_names_map
    )
    self._build_intermediate_ops(
      onnx_graph, ugraph, op_types_cnt, tensor_names_map,
    )
    # find outupt nodes
    distinct_out_ops = set()
    graph_output = set([v.name for v in onnx_graph.output])
    for name, tensor_info in tensor_names_map.items():
      if name in graph_output:
        distinct_out_ops.add(tensor_info.op_name)
    ugraph.output_nodes = list(distinct_out_ops)
    topologic_order_graph(ugraph)
    _PostProcessing.post_process(ugraph)
  
  def _build_param_ops(self, onnx_graph, ugraph, op_types_cnt, tensor_names_map):
    """Find all tensors in initialization list in onnx_graph, normally constants

    Note that this method will update op_types_cnt and tensor_names_map **inplace**
    """
    # find Const ops
    params_dict = {}
    # FIXME: avoid using internal api of other library
    dict_items = _onnx_initializer_to_input_dict_items(onnx_graph.initializer)
    for name, tf_tensor in dict_items:
      params_dict[name] = AttrValueConverter.GenericType(
        value_name='value',
        value=TensorProtoConverter.get_generic_value(
          tf_tensor.op.get_attr('value')
        )
      )
    # build Const ops
    for tensor_name, tensor_value in params_dict.items():
      cnt = op_types_cnt['Const']
      node_name = self._format_node_name(tensor_name, 'Const', cnt)
      op_types_cnt['Const'] += 1
      tensor_names_map[tensor_name] = TensorInfo(
        name=self._format_tensor_name('', node_name, 0),
        op_name=node_name,
        dtype=tensor_value.value.dtype,
        shape=list(tensor_value.value.np_array.shape),
        ugraph=ugraph
      )
      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[tensor_names_map[tensor_name]],
        op_type='Const',
        lib_name='onnx',
        ugraph=ugraph,
        op_attr={
          'value': tensor_value
        }
      )

  def _build_input_ops(self, onnx_graph, ugraph, op_types_cnt, tensor_names_map):
    """Find placeholders
    That is, those ops in the input list but not in initialization list

    Note this method will update inputs **inplace**
    """
    # placeholders
    for value in onnx_graph.input:
      # value: ValueInfoProto
      tensor_name = value.name
      if tensor_name in tensor_names_map:
        # tensor in initializers MAY appears in input, ignore
        continue
      cnt = op_types_cnt['Placeholder']
      node_name = self._format_node_name(tensor_name, 'Placeholder', cnt)
      op_types_cnt['Placeholder'] += 1
      assert value.type.HasField('tensor_type'), 'invalid graph input value'
      tensor_type = value.type.tensor_type
      dtype_str = TensorProto.DataType.Name(tensor_type.elem_type).lower()
      if dtype_str == 'float':
        dtype_str = 'float32'
      dtype = np.dtype(dtype_str)
      shape = [
        dim.dim_value for dim in tensor_type.shape.dim
      ]
      tensor_names_map[tensor_name] = TensorInfo(
        name=self._format_tensor_name('', node_name, 0),
        op_name=node_name,
        dtype=dtype,
        shape=shape,
        ugraph=ugraph,
      )
      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[tensor_names_map[tensor_name]],
        op_type='Placeholder',
        ugraph=ugraph,
        lib_name='onnx',
        op_attr={}
      )

  def _build_intermediate_ops(self, onnx_graph, ugraph, op_types_cnt, tensor_names_map):
    """Build all intermediate nodes, the nodes that is not in neither initialization list nor input list
    """
    # create all outupt tensors
    for node in onnx_graph.node:
      cnt = op_types_cnt[node.op_type]
      node_name = self._format_node_name(node.name, node.op_type, cnt)
      op_types_cnt[node.op_type] += 1
      for i, name in enumerate(node.output):
        tensor_names_map[name] = TensorInfo(
          name=self._format_tensor_name(name, node_name, i),
          op_name=node_name,
          dtype=None,
          shape=None,
          ugraph=ugraph
        )
    # create ops
    for node in onnx_graph.node:
      input_tensors = [
        tensor_names_map[name] for name in node.input
      ]
      output_tensors = [
        tensor_names_map[name] for name in node.output
      ]
      op_attr = {
        attrib_pb.name: _convert_op_attribute(attrib_pb)
        for attrib_pb in node.attribute
      }
      node_name = output_tensors[0].op_name
      OperationInfo(
        name=node_name,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        op_type=node.op_type,
        lib_name='onnx',
        ugraph=ugraph,
        op_attr=op_attr
      )

  def _format_node_name(self, node_name, op_type, op_cnt):
    if node_name == '':
      node_name = '{}_{}'.format(op_type, op_cnt)
    return re.sub(r'[\.:/]', '_', node_name)

  def _format_tensor_name(self, name, node_name, offset):
    if re.match(r'[a-zA-Z][a-zA-Z0-9]*:[0-9]+', name):
      return name
    return '{}:{}'.format(node_name, offset)


class _PostProcessing(object):

  @classmethod
  def post_process(cls, ugraph):
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      handler = getattr(cls, '_handle_{}'.format(op_info.op_type.lower()), lambda op_info: op_info)
      handler(op_info)

  @staticmethod
  def _handle_gemm(op_info):
    ugraph = op_info.ugraph
    output_tensor = op_info.output_tensors[0]
    if output_tensor.dtype is not None and output_tensor.shape is not None:
      return
    input_a, input_w, input_bias = op_info.input_tensors
    a = np.zeros(input_a.shape, dtype=input_a.dtype)
    w = np.zeros(input_w.shape, dtype=input_w.dtype)
    b = np.zeros(input_bias.shape, dtype=input_bias.dtype)
    out = np.matmul(a, w.T) + b
    output_tensor.dtype = out.dtype
    output_tensor.shape = list(out.shape)
    for op in output_tensor.op.output_nodes:
      for i, in_tensor in enumerate(op.input_tensors):
        if in_tensor.name == output_tensor.name:
          op.input_tensors[i] = output_tensor

  @staticmethod
  def _handle_relu(op_info):
    input_tensor = op_info.input_tensors[0]
    op_info.output_tensors[0].dtype = input_tensor.dtype
    op_info.output_tensors[0].shape = input_tensor.shape[:]
  
  @staticmethod
  def _handle_softmax(op_info):
    input_tensor = op_info.input_tensors[0]
    output_tensor = op_info.output_tensors[0]
    if output_tensor.dtype is not None and output_tensor.shape is not None:
      return
    logistics = np.zeros(input_tensor.shape, dtype=input_tensor.dtype)
    out = np.exp(-logistics) * 1e-6
    out /= out.sum(axis=1)
    output_tensor.shape = list(out.shape)
    output_tensor.dtype = out.dtype
