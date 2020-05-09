from utensor_cgen.ir import OperationInfo, TensorInfo
from utensor_cgen.utils import topologic_order_graph

from .api import Legalizer
from .base import LegalizerBase


@Legalizer.register
class OnnxLegalizer(LegalizerBase):
  TARGET = 'onnx'

  def legalize(self, ugraph):
    self._visit_all(ugraph)
    return ugraph

  def _visit_all(self, ugraph):
    for op_info in list(ugraph.ops_info.values()):
      visitor = getattr(
        self,
        '_visit_{}'.format(op_info.op_type.lower()),
        lambda op_info: op_info
      )
      visitor(op_info)
    topologic_order_graph(ugraph)

  # _visit_<op_type> methods will be invoked when an op_info of
  # type <op_type> is encountered during graph traversal
  # ex: _visit_gemm -> will be invoked with an op_info with type Gemm
  def _visit_gemm(self, op_info):
    # op_info.op_type == "Gemm"
    ugraph = op_info.ugraph
    op_info.op_type = 'MatMul'
    tensor_a, tensor_w, tensor_bias = op_info.input_tensors
    out_tensor = TensorInfo(
      name='{}_MatMul:0'.format(op_info.name),
      op_name='{}_MatMul'.format(op_info.name),
      dtype=op_info.output_tensors[0].dtype,
      shape=op_info.output_tensors[0].shape[:],
      ugraph=ugraph,
    )
    OperationInfo(
      name='{}_MatMul'.format(op_info.name),
      input_tensors=[tensor_a, tensor_w],
      output_tensors=[out_tensor],
      op_type='MatMul',
      lib_name=op_info.lib_name,
      ugraph=ugraph,
      op_attr=op_info.op_attr
    )
    add_op = OperationInfo(
      name='{}_AddBias'.format(op_info.name),
      input_tensors=[out_tensor, tensor_bias],
      output_tensors=op_info.output_tensors[:],
      op_type='Add',
      lib_name=op_info.lib_name,
      ugraph=ugraph,
      op_attr={}
    )
    op_info.output_tensors[0].op_name = add_op.name

  def _visit_id(self, op_info):
    # identity op should be op_info.op_type == 'Identity'
    return op_info
