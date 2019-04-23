from __future__ import absolute_import
import os
import six

import numpy as np

from utensor_cgen.frontend.base import Parser
from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.ir.base import TensorInfo, OperationInfo, uTensorGraph
from utensor_cgen.utils import topologic_order_graph

import flatbuffers
from tflite_flatbuffer.Model import Model

@FrontendSelector.register(target_exts=['.tflite'])
class TFLiteParser(Parser):

  @classmethod
  def parse(self, pb_file, output_nodes=None):
    buf = open(pb_file, 'rb').read()
    buf = bytearray(buf)
    model = Model.GetRootAsModel(buf, 0)
    #TODO: version check
    #print(model.Version())

    assert model.SubgraphsLength() == 1
    subgraph = model.Subgraphs(0)

    #table to convert opcode to op-class
    opcode_to_class= []
    for i in range(0, model.OperatorCodesLength()):
        op_code =  model.OperatorCodes(i)
        opcode_to_class.append(op_code)

    #construct lookup objects for builtin ops
    from tflite.BuiltinOperator import BuiltinOperator
    from tflite.BuiltinOptions import BuiltinOptions

    builtin_ops = {v: k for k, v in BuiltinOperator.__dict__.items()}
    op_options = {v: k for k, v in BuiltinOptions.__dict__.items()}
    builtin_option_class = self.get_builtin_option_class()

    ugraph = uTensorGraph(output_nodes=output_nodes,
                        backend="tensorflow") #FIXME: with the proper backend

    #assuming this is the equivalent to ugraph's topological order
    for i in range(0, subgraph.OperatorsLength()):
      op = subgraph.Operators(i)
      opIndex = op.OpcodeIndex()
      op_class = opcode_to_class[opIndex]
      builtin_code = op_class.BuiltinCode()
      op_type = builtin_ops[builtin_code] #op's type in string

      op_attr = dict()

      if(op.CustomOptionsLength() < 1):
        option = builtin_option_class[op_type]
        builtin_data = op.BuiltinOptions()
        option.Init(builtin_data.Bytes, builtin_data.Pos)
        op_attr['option'] = option

        if(op_type == 'FULLY_CONNECTED'):
          from tflite_flatbuffer.FullyConnectedOptionsWeightsFormat import FullyConnectedOptionsWeightsFormat
          w_formats = {v: k for k, v in FullyConnectedOptionsWeightsFormat.__dict__.items()}
          op_attr['weights_format'] = w_formats
        else:
          op_attr['custom_option'] = op.CustomOptionsAsNumpy()

      fb_input_tensors = [subgraph.Tensors(input_idx) for input_idx in op.InputsAsNumpy()]
      in_tensors = [TensorInfo(name=tensor.Name(),
                    ugraph=ugraph,
                    op_name=tensor.op.name, #FIXME: who the fuck knows
                    dtype=self.tensor_type_lookup(tensor.Type()),
                    shape=tensor.ShapeAsNumpy())
                    for tensor in fb_input_tensors]

      fb_output_tensors = [subgraph.Tensors(output_idx) for output_idx in op.OutputsAsNumpy()]
      out_tensors = [TensorInfo(name=tensor.Name(),
              ugraph=ugraph,
              op_name=tensor.op.name, #FIXME: who the fuck knows
              dtype=self.tensor_type_lookup(tensor.Type()),
              shape=tensor.ShapeAsNumpy())
              for tensor in fb_output_tensors]

      node_name = op_type + "_" + i

      op_info = OperationInfo(name=node_name,
        input_tensors=in_tensors,
        output_tensors=out_tensors,
        op_type=op_type,
        backend='tensorflow',  #FIXME: what should this be?
        op_attr=op_attr,
        ugraph=ugraph)

      ugraph.ops_info[node_name] = op_info
    
    topologic_order_graph(ugraph)
    return ugraph

      
  def get_builtin_option_class(self):
    from tflite_flatbuffer.FullyConnectedOptions import FullyConnectedOptions
    from tflite_flatbuffer.ArgMaxOptions import ArgMaxOptions

    table = dict()
    table['FULLY_CONNECTED'] = FullyConnectedOptions()
    table['ARG_MAX'] = ArgMaxOptions()

    return table
  
  def tensor_type_lookup(self, int_type):
      #see TensorType.py
      return -1
