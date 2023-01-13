import os
import re
from collections import defaultdict

import numpy as np

from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.frontend.base import Parser
from utensor_cgen.ir.base import OperationInfo, TensorInfo, uTensorGraph
from utensor_cgen.ir.converter import (AttrValueConverter,
                                       GenericTensorConverterMixin)
from utensor_cgen.legalizer import Legalizer
from utensor_cgen.logger import logger
from utensor_cgen.utils import topologic_order_graph

# schema: https://github.com/tensorflow/tensorflow/blob/a5c0bbb2d15b0708d508f9c930d65d4a584aa338/tensorflow/lite/schema/schema.fbs
from .tflite_flatbuffer.ActivationFunctionType import ActivationFunctionType
from .tflite_flatbuffer.BuiltinOperator import BuiltinOperator
from .tflite_flatbuffer.CustomOptionsFormat import CustomOptionsFormat
from .tflite_flatbuffer.FullyConnectedOptionsWeightsFormat import \
    FullyConnectedOptionsWeightsFormat
from .tflite_flatbuffer.Model import Model

_CUSTOM_OPTION_FORMAT_MAP = {v: k for k, v in CustomOptionsFormat.__dict__.items()}

@FrontendSelector.register(target_exts=[".tflite"])
class TFLiteParser(Parser):
  _TENSOR_NP_TYPE = {
    0: np.dtype("float32"),
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
    # addresseed by index
    tensor_names_map = self._build_tensor_map(fb_model, ugraph)

    self._build_param_ops(fb_model, ugraph, tensor_names_map)
    # find and set input nodes
    self._build_input_ops(fb_model, ugraph, tensor_names_map)
    self._build_intermediate_ops(fb_model, ugraph, tensor_names_map)
    self._set_output_ops(fb_model, ugraph, tensor_names_map)
    self._prepare_quant_params(ugraph)
    topologic_order_graph(ugraph)

  def _build_tensor_map(self, fb_model, ugraph):
    tensor_names_map = {}
    subgraph = self._get_tflm_get_subgraph(fb_model)

    for idx in range(0, subgraph.TensorsLength()):
      tensor = subgraph.Tensors(idx)
      tensor_name = tensor.Name().decode('utf8')
      illegal_chars = ';'
      if any(c in tensor_name for c in illegal_chars):
        logger.warning(
          f'Unexpected character founded in tensor name {tensor_name}, will be replaced or pruned'
        )
      tensor_name = tensor_name.replace(';', '__')
      if tensor_name == "" or tensor_name is None:
        tensor_name = "tensor_" + str(idx)

      dtype = self._TENSOR_NP_TYPE[tensor.Type()]
      attributes = dict()
      quant_params = tensor.Quantization()
      if quant_params is not None and \
        quant_params.ZeroPointLength() and \
        quant_params.ScaleLength():
        attributes["quantization_zeros"] = quant_params.ZeroPointAsNumpy()
        attributes["quantization_scales"] = quant_params.ScaleAsNumpy()

      if isinstance(tensor.ShapeAsNumpy(), np.ndarray):
        shape = tensor.ShapeAsNumpy().tolist()
      elif isinstance(tensor.ShapeAsNumpy(), int):
        logger.warning(f"{tensor.Name().decode('utf8')} is scalar, convert to tensor as shape [1]")
        shape = [1]
      else:
        shape = list(fb_model.Buffers(12).DataAsNumpy().view(dtype).shape)

      tensor_names_map[idx] = TensorInfo(
        name=self._format_tensor_name("", tensor_name, 0),
        op_name="",
        dtype=dtype,
        shape=shape,
        attributes=attributes,
        ugraph=ugraph,
      )
    return tensor_names_map

  def _build_param_ops(self, fb_model, ugraph, tensor_names_map):
    """Const tensors are identified by buffer_index == 0. These tensors are converted to Const Op and added to ugraph
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)

    for idx in range(0, subgraph.TensorsLength()):
      tensor = subgraph.Tensors(idx)
      buffer_index = tensor.Buffer()

      # buffer_index == 0 if intermediate
      if buffer_index == 0:
        continue

      node_name = re.sub(r':\d+', '', tensor_names_map[idx].name) + "_Const"
      dtype = tensor_names_map[idx].dtype

      buffer_array = fb_model.Buffers(buffer_index).DataAsNumpy()
      if isinstance(buffer_array, int):
        continue  # somehow, sometimes, the buffer contains no data, likely to be an intermediate tensor
      buffer_content = fb_model.Buffers(buffer_index).DataAsNumpy().view(dtype).reshape(
        tensor_names_map[idx].shape
      )

      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[tensor_names_map[idx]],
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

      self._set_tensor_node(idx, node_name, tensor_names_map)

  def _build_input_ops(self, fb_model, ugraph, tensor_names_map):
    """Find placeholders
    Attach placeholders to input tensors
    Note this method will update inputs **inplace**
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)
    subgraph_inputs_indexi = subgraph.InputsAsNumpy()
    for index in subgraph_inputs_indexi:
      node_name = tensor_names_map[index].name + "_Placeholder"
      self._set_tensor_node(index, node_name, tensor_names_map)
      OperationInfo(
        name=node_name,
        input_tensors=[],
        output_tensors=[tensor_names_map[index]],
        op_type="Placeholder",
        ugraph=ugraph,
        lib_name="tflm",
        op_attr={},
      )

  def _build_intermediate_ops(self, fb_model, ugraph, tensor_names_map):
    """Build all intermediate nodes
    """
    subgraphs_len = fb_model.SubgraphsLength()
    assert subgraphs_len == 1, "only 1 subgraph is supported"
    subgraph = fb_model.Subgraphs(0)
    for i in range(0, subgraph.OperatorsLength()):
      # topological order, op-index defined by schema
      # BuiltinOperator: https://github.com/tensorflow/tensorflow/blob/031804922d8f4d18b61e3ad077f9f1b69273ff21/tensorflow/lite/schema/schema_v3.fbs#L71
      op = subgraph.Operators(i)
      op_type = TFLiteParser.get_op_type(op, fb_model)

      node_name = str(i) + "_" + op_type

      input_tensor_names = [
        tensor_names_map[input_index] for input_index in op.InputsAsNumpy()
      ]
      output_tensor_names = [
        tensor_names_map[output_index]
        for output_index in op.OutputsAsNumpy()
      ]
      op_attr = _OP_DATA_FUNC_MAP[op_type](op, fb_model)

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
        self._set_tensor_node(tensor_index, node_name, tensor_names_map)
  
  def _set_output_ops(self, fb_model, ugraph, tensor_names_map):
    """identfy output nodes in fb_mdel
    sets output_nodes in ugraph
    Note this method will update ugraph **inplace**
    """
    subgraph = self._get_tflm_get_subgraph(fb_model)
    subgraph_outputs_indexi = subgraph.OutputsAsNumpy()  # tensor indexi
    output_node_names = set()
    for index in subgraph_outputs_indexi:
      output_node_names.add(tensor_names_map[index].op_name)

    ugraph.output_nodes = list(output_node_names)

  def _get_tflm_get_subgraph(self, fb_model):
    subgraphs_len = fb_model.SubgraphsLength()
    assert subgraphs_len == 1, "only 1 subgraph is supported"
    subgraph = fb_model.Subgraphs(0)

    return subgraph

  def _set_tensor_node(self, idx, name, tensor_names_map):
    assert tensor_names_map[idx].op_name == ""
    tensor_names_map[idx].op_name = name

  @staticmethod
  def _prepare_quant_params(ugraph):
    # spec: https://www.tensorflow.org/lite/performance/quantization_spec
    for op_info in ugraph.get_ops_by_type('DepthwiseConv2d'):
      bias = op_info.input_tensors[2]
      if 'quantization_zeros' in bias.attributes:
        zp = bias.attributes['quantization_zeros']
        bias.attributes['quantization_zeros'] = zp.astype(np.dtype('int32'))
    for op_info in ugraph.get_ops_by_type('FullyConnected'):
      bias = op_info.input_tensors[2]
      if 'quantization_zeros' in bias.attributes:
        zp = bias.attributes['quantization_zeros']
        bias.attributes['quantization_zeros'] = zp.astype(np.dtype('int32'))
    for op_info in ugraph.get_ops_by_type('Conv2d'):
      bias = op_info.input_tensors[2]
      if 'quantization_zeros' in bias.attributes:
        zp = bias.attributes['quantization_zeros']
        bias.attributes['quantization_zeros'] = zp.astype(np.dtype('int32'))

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

  @classmethod
  def get_op_type(cls, op, fb_model):
    local_op_code = op.OpcodeIndex()
    global_op_code = fb_model.OperatorCodes(local_op_code)
    builtin_op_code = global_op_code.BuiltinCode()
    return cls._BUILTIN_OPS[builtin_op_code]

# helper functions for parsing op data (will be stored in op_attr)
def default_op_data(op, fb_mdel):
  op_type = TFLiteParser.get_op_type(op, fb_mdel)
  logger.warning('the op data parser is missing for %s', op_type)
  return {}

_OP_DATA_FUNC_MAP = defaultdict(lambda: default_op_data)

def _register_op_data_func(op_type):
  def register(func):
    _OP_DATA_FUNC_MAP[op_type] = func
    return func
  return register

def _class_option2str(obj, idx):
  names_lookup = {v: k for k, v in obj.__dict__.items()}
  name = names_lookup[idx]
  return str(idx) + " (" + name + ")"

@_register_op_data_func("FULLY_CONNECTED")
def fully_connected_op_data(op, fb_mdel):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.FullyConnectedOptions import FullyConnectedOptions

    option = FullyConnectedOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["FusedActivationFunction"] = _class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
    option_dict["w_formats"] = _class_option2str(
      FullyConnectedOptionsWeightsFormat, option.WeightsFormat()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict

@_register_op_data_func("DEPTHWISE_CONV_2D")
def depthwise_conv2d_op_data(op, fb_mdel):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.DepthwiseConv2DOptions import \
        DepthwiseConv2DOptions

    option = DepthwiseConv2DOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["Padding"] = option.Padding()
    option_dict["StrideW"] = option.StrideW()
    option_dict["StrideH"] = option.StrideH()
    option_dict["DepthMultiplier"] = option.DepthMultiplier()
    option_dict["FusedActivationFunction"] = _class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
    option_dict["DilationWFactor"] = option.DilationWFactor()
    option_dict["DilationHFactor"] = option.DilationHFactor()

  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

@_register_op_data_func("CONV_2D")
def conv_2d_op_data(op, fb_model):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.Conv2DOptions import Conv2DOptions

    option = Conv2DOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["Padding"] = option.Padding()
    option_dict["StrideW"] = option.StrideW()
    option_dict["StrideH"] = option.StrideH()
    option_dict["FusedActivationFunction"] = _class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
    option_dict["DilationWFactor"] = option.DilationWFactor()
    option_dict["DilationHFactor"] = option.DilationHFactor()
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

@_register_op_data_func("RESHAPE")
def reshape_op_data(op, fb_mdel):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.ReshapeOptions import ReshapeOptions

    option = ReshapeOptions()
    builtin_data = op.BuiltinOptions()
    if builtin_data is None:
      option_dict["new_shape"] = list()
    else:
      option.Init(builtin_data.Bytes, builtin_data.Pos)
      option_dict["new_shape"] = list(option.NewShapeAsNumpy())
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

@_register_op_data_func("DEQUANTIZE")
def dequantize_op_data(op, fb_mdel):
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

@_register_op_data_func("QUANTIZE")
def quantize_op_data(op, fb_mdel):
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

@_register_op_data_func("AVG_POOL_2D")
@_register_op_data_func("MIN_POOL_2D")
@_register_op_data_func("MAX_POOL_2D")
def pool2d_op_data(op, fb_mdel):
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
    option_dict["FusedActivationFunction"] = _class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()

  return option_dict

@_register_op_data_func("ARG_MAX")
def argmax_op_data(op, fb_model):
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

@_register_op_data_func("TRANSPOSE")
def transpose_op_data(op, fb_model):
  option_dict = {}
  from .tflite_flatbuffer.TransposeOptions import TransposeOptions

  # no filed declared in the fbs file for TransposeOptions
  # skipping here
  # this function is here just for silencing the warning msg
  return option_dict

@_register_op_data_func("STRIDED_SLICE")
def stride_slice_op_data(op, fb_model):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.StridedSliceOptions import StridedSliceOptions

    option = StridedSliceOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["begin_mask"] = option.BeginMask()
    option_dict["end_mask"] = option.EndMask()
    option_dict["ellipsis_mask"] = option.EllipsisMask()
    option_dict["shrink_axis_mask"] = option.ShrinkAxisMask()
    option_dict["new_axis_mask"] = option.NewAxisMask()
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict

@_register_op_data_func("TANH")
def dummy_op_data(op, fb_model):
  """
  dummy func
  """
  return {}

@_register_op_data_func("CONCATENATION")
def concat_op_data(op, fb_model):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.ConcatenationOptions import ConcatenationOptions

    option = ConcatenationOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["axis"] = option.Axis()
    option_dict["FusedActivationFunction"] = _class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict

@_register_op_data_func("EXPAND_DIMS")
def expand_dims_op_data(op, fb_model):
  """
  dummy, just in case if there is anything need to be parsed here
  """
  from .tflite_flatbuffer.ExpandDimsOptions import ExpandDimsOptions

  return {}

@_register_op_data_func("DIV")
def div_op_data(op, fb_model):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.DivOptions import DivOptions

    option = DivOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["FusedActivationFunction"] = _class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict


@_register_op_data_func("MUL")
def mul_op_data(op, fb_model):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.MulOptions import MulOptions

    option = MulOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["FusedActivationFunction"] = _class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict

@_register_op_data_func("ADD")
def add_op_data(op, fb_model):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.AddOptions import AddOptions

    option = AddOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["FusedActivationFunction"] = _class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict

@_register_op_data_func("SUB")
def sub_op_data(op, fb_model):
  option_dict = {}
  if op.CustomOptionsLength() < 1:
    from .tflite_flatbuffer.SubOptions import SubOptions

    option = SubOptions()
    builtin_data = op.BuiltinOptions()
    option.Init(builtin_data.Bytes, builtin_data.Pos)
    option_dict["FusedActivationFunction"] = _class_option2str(
      ActivationFunctionType, option.FusedActivationFunction()
    )
  else:
    option_dict[
      _CUSTOM_OPTION_FORMAT_MAP[op.CustomOptionsFormat()]
    ] = op.CustomOptionsAsNumpy()
  return option_dict
