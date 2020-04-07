from __future__ import absolute_import
import os
import six
import re

import numpy as np

from utensor_cgen.frontend.base import Parser
from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.ir.base import TensorInfo, OperationInfo, uTensorGraph
from utensor_cgen.utils import topologic_order_graph

import flatbuffers
from .tflite_flatbuffer.Model import Model

tensor_np_type = dict()
tensor_np_type[0] = np.float32
tensor_np_type[1] = np.float16
tensor_np_type[2] = np.int32
tensor_np_type[3] = np.uint8
tensor_np_type[4] = np.uint64
tensor_np_type[5] = np.ubyte #FIXME: supposed to be string
tensor_np_type[6] = np.bool
tensor_np_type[7] = np.int16
tensor_np_type[8] = np.cdouble
tensor_np_type[9] = np.int8

from .tflite_flatbuffer.BuiltinOperator import BuiltinOperator
builtin_ops = {v: k for k, v in BuiltinOperator.__dict__.items()}

@FrontendSelector.register(target_exts=['.tflite'])
class TFLiteParser(Parser):

  def parse(self, tflite_file, output_nodes=None):
    graph_name, _ = os.path.splitext(tflite_file)
    buf = open(tflite_file, 'rb').read()
    buf = bytearray(buf)
    fb_model = Model.GetRootAsModel(buf, 0)

    ugraph = uTensorGraph(
      name=graph_name,
      output_nodes=[],
      lib_name='tflite',
      ops_info={},
    )

    #print("TF Lite Parser")

    self._build_graph(fb_model, ugraph)

    return ugraph

  def _build_graph(self, fb_model, ugraph):
    self.tensor_names_map = {} #addresseed by indexi
    self._build_tensor_map(fb_model, ugraph)

    self._build_param_ops(fb_model, ugraph)
    #find and set input nodes
    self._build_input_ops(fb_model, ugraph)
    self._build_intermediate_ops(fb_model, ugraph)
    self._set_output_ops(fb_model, ugraph)

    topologic_order_graph(ugraph)

  def _set_output_ops(self, fb_model, ugraph):
    """identfy output nodes in fb_mdel
    sets output_nodes in ugraph
    Note this method will update ugraph **inplace**
    """
    subgraph_outputs_indexi = fb_model.OutputsAsNumpy() #tensor indexi
    output_node_names = set()
    for index in subgraph_outputs_indexi:
      output_node_names.add(self.tensor_names_map[index].op_name)

    ugraph.output_nodes = list(output_node_names)

  def _build_tensor_map(self, fb_model, ugraph):
    subgraph = self._get_tflm_get_subgraph(fb_model)

    for idx in range(0, subgraph.TensorsLength()):
      tensor = subgraph.Tensors(idx)

      tensor_name = tensor.Name()
      if tensor_name is '' or None:
        tensor_name = 'tensor_' + str(idx)
  
      dtype=tensor_np_type[tensor.Type()]

      attributes = dict()
      attributes['quantizationParam'] = tensor.Quantization()

      self.tensor_names_map[idx] = TensorInfo(
        name=self._format_tensor_name('', tensor_name, 0),
        op_name="",
        dtype=dtype,
        shape=tensor.ShapeAsNumpy(),
        attributes=attributes,
        ugraph=ugraph
      )

      # 0 if intermediate
      #buffer_index = tensor.Buffer()
      #buffer_content = model.Buffers(buffer_index).DataAsNumpy().astype(dtype)

      #tensor.Type()

  def _build_param_ops(self, fb_model, ugraph):
    """Find all tensors in initialization list in onnx_graph, normally constants
    Note that this method will update op_types_cnt and tensor_names_map **inplace**
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)

    for idx in range(0, subgraph.TensorsLength()):
      tensor = subgraph.Tensors(idx)
      buffer_index = tensor.Buffer()

      if buffer_index == 0:
        continue

      node_name = self.tensor_names_map[idx].name + "_Const"
      dtype = self.tensor_names_map[idx].dtype

      buffer_content = fb_model.Buffers(buffer_index).DataAsNumpy().astype(dtype)

      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[self.tensor_names_map[idx]],
        op_type='Const',
        lib_name='tflm',
        ugraph=ugraph,
        op_attr={
          'value': buffer_content
        }
      )

      self._set_tensor_node(idx, node_name)

  def _build_input_ops(self, fb_model, ugraph):
    """Find placeholders
    Attach placeholders to input tensors
    Note this method will update inputs **inplace**
    """
    subgraph_inputs_indexi = fb_model.InputsAsNumpy()
    for index in subgraph_inputs_indexi:
      node_name = self.tensor_names_map[index].name + "_Placeholder"
      self._set_tensor_node(index, node_name)
      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[self.tensor_names_map[index]],
        op_type='Placeholder',
        ugraph=ugraph,
        lib_name='tflm',
        op_attr={}
      )

  def _build_intermediate_ops(self, fb_model, ugraph):
    """Build all intermediate nodes, the nodes that is not in neither initialization list nor input list
    """
    subgraphs_len = fb_model.SubgraphsLength()
    assert subgraphs_len == 1, "only 1 subgraph is supported"
    subgraph = fb_model.Subgraphs(0)
    for i in range(0, subgraph.OperatorsLength()):
      #topological order, op-index defined by schema
      #BuiltinOperator: https://github.com/tensorflow/tensorflow/blob/031804922d8f4d18b61e3ad077f9f1b69273ff21/tensorflow/lite/schema/schema_v3.fbs#L71
      op = subgraph.Operators(i)
      local_op_code = op.OpcodeIndex()
      global_op_code = fb_model.OperatorCodes(local_op_code)
      builtinOperator_code = global_op_code.BuiltinCode()
      op_type = builtin_ops[builtinOperator_code]

      node_name = str(i) + "_" + op_type

      input_tensor_names = [self.tensor_names_map[input_index].name for input_index in op.InputsAsNumpy()]
      output_tensor_names = [self.tensor_names_map[output_index].name for output_index in op.OutputsAsNumpy()]

      OperationInfo(
        name=node_name,
        input_tensors=input_tensor_names,
        output_tensors=output_tensor_names,
        op_type=op_type,
        ugraph=ugraph,
        lib_name='tflm',
        op_attr={}
      )
      
      for tensor_index in op.OutputsAsNumpy():
        self._set_tensor_node(tensor_index, node_name)

  def _get_tflm_get_subgraph(self, fb_model):
    subgraphs_len = fb_model.SubgraphsLength()
    assert subgraphs_len == 1, "only 1 subgraph is supported"
    subgraph = fb_model.Subgraphs(0)

    return subgraph

  def _set_tensor_node(self, idx, name):
    assert self.tensor_names_map[idx].op_name != ""
    self.tensor_names_map[idx].op_name = name

  def _format_node_name(self, node_name, op_type, op_cnt):
    if node_name == '':
      node_name = '{}_{}'.format(op_type, op_cnt)
    return re.sub(r'[\.:/]', '_', node_name)

  def _format_tensor_name(self, name, node_name, offset):
    if re.match(r'[a-zA-Z][a-zA-Z0-9]*:[0-9]+', name):
      return name
    return '{}:{}'.format(node_name, offset)
