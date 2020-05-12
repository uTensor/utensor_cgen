import os
import re

import numpy as np

from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.frontend.base import Parser
from utensor_cgen.ir.base import OperationInfo, TensorInfo, uTensorGraph
from utensor_cgen.ir.converter import (AttrValueConverter,
                                       GenericTensorConverterMixin)
from utensor_cgen.legalizer import Legalizer
from utensor_cgen.utils import topologic_order_graph

from .tflite_flatbuffer.ActivationFunctionType import ActivationFunctionType
from .tflite_flatbuffer.BuiltinOperator import BuiltinOperator
from .tflite_flatbuffer.CustomOptionsFormat import CustomOptionsFormat
from .tflite_flatbuffer.FullyConnectedOptionsWeightsFormat import \
    FullyConnectedOptionsWeightsFormat
from .tflite_flatbuffer.Model import Model

_CUSTOM_OPTION_FORMAT_MAP = {v: k for k, v in CustomOptionsFormat.__dict__.items()}

def class_option2str(obj, idx):
  names_lookup = {v: k for k, v in obj.__dict__.items()}
  name = names_lookup[idx]
  return str(idx) + " (" + name + ")"

def fully_connected_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.FullyConnectedOptions import FullyConnectedOptions

    option = FullyConnectedOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["FusedActivationFunction"] = class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
    option_dict["w_formats"] = class_option2str(
      FullyConnectedOptionsWeightsFormat, option.WeightsFormat()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict

def depthwise_conv2d_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.DepthwiseConv2DOptions import DepthwiseConv2DOptions

    option = DepthwiseConv2DOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["Padding"] = option.Padding()
    option_dict["StrideW"] = option.StrideW()
    option_dict["StrideH"] = option.StrideH()
    option_dict["DepthMultiplier"] = option.DepthMultiplier()
    option_dict["FusedActivationFunction"] = class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
    option_dict["DilationWFactor"] = option.DilationWFactor()
    option_dict["DilationHFactor"] = option.DilationHFactor()

  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def conv2d_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.Conv2DOptions import Conv2DOptions

    option = Conv2DOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["Padding"] = option.Padding()
    option_dict["StrideW"] = option.StrideW()
    option_dict["StrideH"] = option.StrideH()
    option_dict["FusedActivationFunction"] = class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
    option_dict["DilationWFactor"] = option.DilationWFactor()
    option_dict["DilationHFactor"] = option.DilationHFactor()

  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def reshape_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.ReshapeOptions import ReshapeOptions

    option = ReshapeOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["new_shape"] = list(option.NewShapeAsNumpy())
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def reducer_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.ReducerOptions import ReducerOptions

    option = ReducerOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["keep_dims"] = option.KeepDims()
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def dequantize_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.DequantizeOptions import DequantizeOptions

    option = DequantizeOptions()
    builtin_data = op.BuiltinOptions()
    if builtin_data is None:
      return option_dict
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict['builtin'] = option
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def quantize_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.QuantizeOptions import QuantizeOptions

    option = QuantizeOptions()
    builtin_data = op.BuiltinOptions()
    if builtin_data is None:
      return option_dict
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict['builtin'] = option
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def pool2d_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.Pool2DOptions import Pool2DOptions

    option = Pool2DOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["Padding"] = option.Padding()
    option_dict["StrideW"] = option.StrideW()
    option_dict["StrideH"] = option.StrideH()
    option_dict["FilterWidth"] = option.FilterWidth()
    option_dict["FilterHeight"] = option.FilterHeight()
    option_dict["FusedActivationFunction"] = class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def argmax_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.ArgMaxOptions import ArgMaxOptions

    option = ArgMaxOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["OutputType"] = option.OutputType()
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def mul_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.MulOptions import MulOptions

    option = MulOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    #option_dict["OutputType"] = option.OutputType()
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def sub_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.SubOptions import SubOptions

    option = SubOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    #option_dict["OutputType"] = option.OutputType()
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

def softmax_op_data(op):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.SoftmaxOptions import SoftmaxOptions

    option = SoftmaxOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["Beta"] = option.Beta()
    #option_dict["OutputType"] = option.OutputType()
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict


_OP_DATA_FUNC_MAP                                 = dict()
_OP_DATA_FUNC_MAP["ADD"]                          = None
_OP_DATA_FUNC_MAP["AVERAGE_POOL_2D"]              = None
_OP_DATA_FUNC_MAP["CONCATENATION"]                = None
_OP_DATA_FUNC_MAP["CONV_2D"]                      = conv2d_op_data
_OP_DATA_FUNC_MAP["DEPTHWISE_CONV_2D"]            = depthwise_conv2d_op_data
_OP_DATA_FUNC_MAP["DEQUANTIZE"]                   = dequantize_op_data
_OP_DATA_FUNC_MAP["EMBEDDING_LOOKUP"]             = None
_OP_DATA_FUNC_MAP["FLOOR"]                        = None
_OP_DATA_FUNC_MAP["FULLY_CONNECTED"]              = fully_connected_op_data
_OP_DATA_FUNC_MAP["HASHTABLE_LOOKUP"]             = None
_OP_DATA_FUNC_MAP["L2_NORMALIZATION"]             = None
_OP_DATA_FUNC_MAP["L2_POOL_2D"]                   = None
_OP_DATA_FUNC_MAP["LOCAL_RESPONSE_NORMALIZATION"] = None
_OP_DATA_FUNC_MAP["LOGISTIC"]                     = None
_OP_DATA_FUNC_MAP["LSH_PROJECTION"]               = None
_OP_DATA_FUNC_MAP["LSTM"]                         = None
_OP_DATA_FUNC_MAP["MAX_POOL_2D"]                  = pool2d_op_data
_OP_DATA_FUNC_MAP["MUL"]                          = mul_op_data
_OP_DATA_FUNC_MAP["RELU"]                         = None
_OP_DATA_FUNC_MAP["RELU_N1_TO_1"]                 = None
_OP_DATA_FUNC_MAP["RELU6"]                        = None
_OP_DATA_FUNC_MAP["RESHAPE"]                      = reshape_op_data
_OP_DATA_FUNC_MAP["RESIZE_BILINEAR"]              = None
_OP_DATA_FUNC_MAP["RNN"]                          = None
_OP_DATA_FUNC_MAP["SOFTMAX"]                      = softmax_op_data
_OP_DATA_FUNC_MAP["SPACE_TO_DEPTH"]               = None
_OP_DATA_FUNC_MAP["SVDF"]                         = None
_OP_DATA_FUNC_MAP["TANH"]                         = None
_OP_DATA_FUNC_MAP["CONCAT_EMBEDDINGS"]            = None
_OP_DATA_FUNC_MAP["SKIP_GRAM"]                    = None
_OP_DATA_FUNC_MAP["CALL"]                         = None
_OP_DATA_FUNC_MAP["CUSTOM"]                       = None
_OP_DATA_FUNC_MAP["EMBEDDING_LOOKUP_SPARSE"]      = None
_OP_DATA_FUNC_MAP["PAD"]                          = None
_OP_DATA_FUNC_MAP["UNIDIRECTIONAL_SEQUENCE_RNN"]  = None
_OP_DATA_FUNC_MAP["GATHER"]                       = None
_OP_DATA_FUNC_MAP["BATCH_TO_SPACE_ND"]            = None
_OP_DATA_FUNC_MAP["SPACE_TO_BATCH_ND"]            = None
_OP_DATA_FUNC_MAP["TRANSPOSE"]                    = None
_OP_DATA_FUNC_MAP["MEAN"]                         = reducer_op_data
_OP_DATA_FUNC_MAP["SUB"]                          = sub_op_data
_OP_DATA_FUNC_MAP["DIV"]                          = None
_OP_DATA_FUNC_MAP["SQUEEZE"]                      = None
_OP_DATA_FUNC_MAP["UNIDIRECTIONAL_SEQUENCE_LSTM"] = None
_OP_DATA_FUNC_MAP["STRIDED_SLICE"]                = None
_OP_DATA_FUNC_MAP["BIDIRECTIONAL_SEQUENCE_RNN"]   = None
_OP_DATA_FUNC_MAP["EXP"]                          = None
_OP_DATA_FUNC_MAP["TOPK_V2"]                      = None
_OP_DATA_FUNC_MAP["SPLIT"]                        = None
_OP_DATA_FUNC_MAP["LOG_SOFTMAX"]                  = None
_OP_DATA_FUNC_MAP["DELEGATE"]                     = None
_OP_DATA_FUNC_MAP["BIDIRECTIONAL_SEQUENCE_LSTM"]  = None
_OP_DATA_FUNC_MAP["CAST"]                         = None
_OP_DATA_FUNC_MAP["PRELU"]                        = None
_OP_DATA_FUNC_MAP["MAXIMUM"]                      = None
_OP_DATA_FUNC_MAP["ARG_MAX"]                      = argmax_op_data
_OP_DATA_FUNC_MAP["MINIMUM"]                      = None
_OP_DATA_FUNC_MAP["LESS"]                         = None
_OP_DATA_FUNC_MAP["NEG"]                          = None
_OP_DATA_FUNC_MAP["PADV2"]                        = None
_OP_DATA_FUNC_MAP["GREATER"]                      = None
_OP_DATA_FUNC_MAP["GREATER_EQUAL"]                = None
_OP_DATA_FUNC_MAP["LESS_EQUAL"]                   = None
_OP_DATA_FUNC_MAP["SELECT"]                       = None
_OP_DATA_FUNC_MAP["SLICE"]                        = None
_OP_DATA_FUNC_MAP["SIN"]                          = None
_OP_DATA_FUNC_MAP["TRANSPOSE_CONV"]               = None
_OP_DATA_FUNC_MAP["SPARSE_TO_DENSE"]              = None
_OP_DATA_FUNC_MAP["TILE"]                         = None
_OP_DATA_FUNC_MAP["EXPAND_DIMS"]                  = None
_OP_DATA_FUNC_MAP["EQUAL"]                        = None
_OP_DATA_FUNC_MAP["NOT_EQUAL"]                    = None
_OP_DATA_FUNC_MAP["LOG"]                          = None
_OP_DATA_FUNC_MAP["SUM"]                          = None
_OP_DATA_FUNC_MAP["SQRT"]                         = None
_OP_DATA_FUNC_MAP["RSQRT"]                        = None
_OP_DATA_FUNC_MAP["SHAPE"]                        = None
_OP_DATA_FUNC_MAP["POW"]                          = None
_OP_DATA_FUNC_MAP["ARG_MIN"]                      = None
_OP_DATA_FUNC_MAP["FAKE_QUANT"]                   = None
_OP_DATA_FUNC_MAP["REDUCE_PROD"]                  = None
_OP_DATA_FUNC_MAP["REDUCE_MAX"]                   = None
_OP_DATA_FUNC_MAP["PACK"]                         = None
_OP_DATA_FUNC_MAP["LOGICAL_OR"]                   = None
_OP_DATA_FUNC_MAP["ONE_HOT"]                      = None
_OP_DATA_FUNC_MAP["LOGICAL_AND"]                  = None
_OP_DATA_FUNC_MAP["LOGICAL_NOT"]                  = None
_OP_DATA_FUNC_MAP["UNPACK"]                       = None
_OP_DATA_FUNC_MAP["REDUCE_MIN"]                   = None
_OP_DATA_FUNC_MAP["FLOOR_DIV"]                    = None
_OP_DATA_FUNC_MAP["REDUCE_ANY"]                   = None
_OP_DATA_FUNC_MAP["SQUARE"]                       = None
_OP_DATA_FUNC_MAP["ZEROS_LIKE"]                   = None
_OP_DATA_FUNC_MAP["FILL"]                         = None
_OP_DATA_FUNC_MAP["FLOOR_MOD"]                    = None
_OP_DATA_FUNC_MAP["RANGE"]                        = None
_OP_DATA_FUNC_MAP["RESIZE_NEAREST_NEIGHBOR"]      = None
_OP_DATA_FUNC_MAP["LEAKY_RELU"]                   = None
_OP_DATA_FUNC_MAP["SQUARED_DIFFERENCE"]           = None
_OP_DATA_FUNC_MAP["MIRROR_PAD"]                   = None
_OP_DATA_FUNC_MAP["ABS"]                          = None
_OP_DATA_FUNC_MAP["SPLIT_V"]                      = None
_OP_DATA_FUNC_MAP["UNIQUE"]                       = None
_OP_DATA_FUNC_MAP["CEIL"]                         = None
_OP_DATA_FUNC_MAP["REVERSE_V2"]                   = None
_OP_DATA_FUNC_MAP["ADD_N"]                        = None
_OP_DATA_FUNC_MAP["GATHER_ND"]                    = None
_OP_DATA_FUNC_MAP["COS"]                          = None
_OP_DATA_FUNC_MAP["WHERE"]                        = None
_OP_DATA_FUNC_MAP["RANK"]                         = None
_OP_DATA_FUNC_MAP["ELU"]                          = None
_OP_DATA_FUNC_MAP["REVERSE_SEQUENCE"]             = None
_OP_DATA_FUNC_MAP["MATRIX_DIAG"]                  = None
_OP_DATA_FUNC_MAP["QUANTIZE"]                     = quantize_op_data
_OP_DATA_FUNC_MAP["MATRIX_SET_DIAG"]              = None


@FrontendSelector.register(target_exts=[".tflite"])
class TFLiteParser(Parser):
  _TENSOR_NP_TYPE = {
    0:np.dtype("float32"),
    1: np.dtype("float16"),
    2: np.dtype("int32"),
    3: np.dtype("uint8"),
    4: np.dtype("uint64"),
    5: np.dtype("str"),
    6: np.dtype("bool"),
    7: np.dtype("int16"),
    8: np.dtype("cdouble"),
    9: np.dtype("int8"),
  }
  _BUILTIN_OPS = {v: k for k, v in BuiltinOperator.__dict__.items()}

  def parse(self, tflite_file, output_nodes=None, model_name=None):
    if output_nodes is None:
      output_nodes = []
    if model_name:
      graph_name = model_name
    else:
      graph_name, _ = os.path.splitext(
        os.path.basename(tflite_file)
      )
    with open(tflite_file, "rb") as fid:
      buf = bytearray(fid.read())
    fb_model = Model.GetRootAsModel(buf, 0)

    ugraph = uTensorGraph(
      name=graph_name,
      output_nodes=output_nodes,
      lib_name="tflite",
      ops_info={},
    )
    self._build_graph(fb_model, ugraph)
    ugraph = Legalizer.legalize(ugraph)
    return ugraph

  def _build_graph(self, fb_model, ugraph):
    self.tensor_names_map = {}  # addresseed by indexi
    self._build_tensor_map(fb_model, ugraph)

    self._build_param_ops(fb_model, ugraph)
    # find and set input nodes
    self._build_input_ops(fb_model, ugraph)
    self._build_intermediate_ops(fb_model, ugraph)
    self._set_output_ops(fb_model, ugraph)

    topologic_order_graph(ugraph)

  def _set_output_ops(self, fb_model, ugraph):
    """identfy output nodes in fb_mdel
    sets output_nodes in ugraph
    Note this method will update ugraph **inplace**
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)
    subgraph_outputs_indexi = subgraph.OutputsAsNumpy()  # tensor indexi
    output_node_names = set()
    for index in subgraph_outputs_indexi:
      output_node_names.add(self.tensor_names_map[index].op_name)

    ugraph.output_nodes = list(output_node_names)

  def _build_tensor_map(self, fb_model, ugraph):
    subgraph = self._get_tflm_get_subgraph(fb_model)

    for idx in range(0, subgraph.TensorsLength()):
      tensor = subgraph.Tensors(idx)
      tensor_name = tensor.Name().decode('utf8')
      if tensor_name is "" or None:
        tensor_name = "tensor_" + str(idx)
      
      dtype = self._TENSOR_NP_TYPE[tensor.Type()]
      attributes = dict()
      quant_params = tensor.Quantization()
      if quant_params is not None and quant_params.ZeroPointAsNumpy() and quant_params.ScaleAsNumpy():
        zp = quant_params.ZeroPointAsNumpy()
        if zp.dtype == np.dtype('<i8'):
          zp = zp.astype('int8')
        else:
          zp = zp.astype('uint8')
        attributes["quantization_zeros"] = zp
        attributes["quantization_scales"] = quant_params.ScaleAsNumpy()

      if isinstance(tensor.ShapeAsNumpy(), np.ndarray):
        shape = tensor.ShapeAsNumpy().tolist()
      else:
        shape = list(fb_model.Buffers(12).DataAsNumpy().shape)

      self.tensor_names_map[idx] = TensorInfo(
        name=self._format_tensor_name("", tensor_name, 0),
        op_name="",
        dtype=dtype,
        shape=shape,
        attributes=attributes,
        ugraph=ugraph,
      )

  def _build_param_ops(self, fb_model, ugraph):
    """Const tensors are identified by buffer_index == 0. These tensors are converted to Const Op and added to ugraph
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)

    for idx in range(0, subgraph.TensorsLength()):
      tensor = subgraph.Tensors(idx)
      buffer_index = tensor.Buffer()

      # buffer_index == 0 if intermediate
      if buffer_index == 0:
        continue

      node_name = re.sub(r':\d+', '', self.tensor_names_map[idx].name) + "_Const"
      dtype = self.tensor_names_map[idx].dtype

      buffer_array = fb_model.Buffers(buffer_index).DataAsNumpy()
      if isinstance(buffer_array, int):
        continue  # somehow, sometimes, the buffer contains no data, likely to be an intermediate tensor
      buffer_content = fb_model.Buffers(buffer_index).DataAsNumpy().view(dtype).reshape(
        self.tensor_names_map[idx].shape
      )

      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[self.tensor_names_map[idx]],
        op_type="Const",
        lib_name="tflm",
        ugraph=ugraph,
        op_attr={
          "value": AttrValueConverter.GenericType(
            value_name="tensor",
            value=GenericTensorConverterMixin.GenericType(
              np_array=buffer_content
            ),
          )
        },
      )

      self._set_tensor_node(idx, node_name)

  def _build_input_ops(self, fb_model, ugraph):
    """Find placeholders
    Attach placeholders to input tensors
    Note this method will update inputs **inplace**
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)
    subgraph_inputs_indexi = subgraph.InputsAsNumpy()
    for index in subgraph_inputs_indexi:
      node_name = self.tensor_names_map[index].name + "_Placeholder"
      self._set_tensor_node(index, node_name)
      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[self.tensor_names_map[index]],
        op_type="Placeholder",
        ugraph=ugraph,
        lib_name="tflm",
        op_attr={},
      )

  def _build_intermediate_ops(self, fb_model, ugraph):
    """Build all intermediate nodes
    """
    subgraphs_len = fb_model.SubgraphsLength()
    assert subgraphs_len == 1, "only 1 subgraph is supported"
    subgraph = fb_model.Subgraphs(0)
    for i in range(0, subgraph.OperatorsLength()):
      # topological order, op-index defined by schema
      # BuiltinOperator: https://github.com/tensorflow/tensorflow/blob/031804922d8f4d18b61e3ad077f9f1b69273ff21/tensorflow/lite/schema/schema_v3.fbs#L71
      op = subgraph.Operators(i)
      local_op_code = op.OpcodeIndex()
      global_op_code = fb_model.OperatorCodes(local_op_code)
      builtin_op_code = global_op_code.BuiltinCode()
      op_type = self._BUILTIN_OPS[builtin_op_code]

      node_name = str(i) + "_" + op_type

      input_tensor_names = [
        self.tensor_names_map[input_index] for input_index in op.InputsAsNumpy()
      ]
      output_tensor_names = [
        self.tensor_names_map[output_index]
        for output_index in op.OutputsAsNumpy()
      ]

      #import pdb; pdb.set_trace()
      op_attr = _OP_DATA_FUNC_MAP[op_type](op)

      OperationInfo(
        name=node_name,
        input_tensors=input_tensor_names,
        output_tensors=output_tensor_names,
        op_type=self._format_op_type(op_type),
        ugraph=ugraph,
        lib_name="tflm",
        op_attr=op_attr,
      )

      for tensor_index in op.OutputsAsNumpy():
        self._set_tensor_node(tensor_index, node_name)

  def _get_tflm_get_subgraph(self, fb_model):
    subgraphs_len = fb_model.SubgraphsLength()
    assert subgraphs_len == 1, "only 1 subgraph is supported"
    subgraph = fb_model.Subgraphs(0)

    return subgraph

  def _set_tensor_node(self, idx, name):
    assert self.tensor_names_map[idx].op_name == ""
    self.tensor_names_map[idx].op_name = name

  def _format_node_name(self, node_name, op_type, op_cnt):
    if node_name == "":
      node_name = "{}_{}".format(op_type, op_cnt)
    return re.sub(r"[\.:/]", "_", node_name)

  def _format_tensor_name(self, name, node_name, offset):
    if re.match(r"[a-zA-Z][a-zA-Z0-9]*:[0-9]+", name):
      return name
    return "{}:{}".format(node_name, offset)

  def _format_op_type(self, op_type):
    return ''.join(map(lambda s: s.capitalize(), op_type.split('_')))
