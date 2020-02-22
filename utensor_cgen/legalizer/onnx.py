from utensor_cgen.ir import OperationInfo, TensorInfo
from utensor_cgen.utils import topologic_order_graph

from .base import LegalizerBase


class OnnxLegalizer(LegalizerBase):
  TARGET = 'onnx'

  def legalize(self, ugraph):
    self._visit(ugraph)
    return ugraph

  def _visit(self, ugraph):
    for op_info in list(ugraph.ops_info.values()):
      visitor = getattr(
        self,
        '_visit_{}'.format(op_info.op_type.lower()),
        lambda op_info: op_info
      )
      visitor(op_info)
    topologic_order_graph(ugraph)

  def _visit_gemm(self, op_info):
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
