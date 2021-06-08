from utensor_cgen.logger import logger
from utensor_cgen.utils import topologic_order_graph


def _hotfix_reshape(ugraph):
  op_names_to_remove = set()
  for op in ugraph.get_ops_by_type("ReshapeOperator"):
    new_shape = op.op_attr.get("new_shape", None)
    if not new_shape:
      logger.warning(f'{op.name} has no new_shape as its attributes, using the second input tensor as new shape instead')
      tensor_new_shape = op.input_tensors.pop(1)
      op.n_inputs -= 1
      ori_new_shape = tensor_new_shape.op.op_attr['value'].value.np_array.tolist()
      new_shape_ = []
      has_neg = False
      for s in ori_new_shape:
        if s < 0:
          has_neg = True
          s = abs(s)
        new_shape_.append(s)
      if has_neg:
        logger.warning(f"implicit convert negative shape of reshape op to positive: {ori_new_shape} -> {new_shape_}")
      op.op_attr["new_shape"] = new_shape_
      op_names_to_remove.add(tensor_new_shape.op.name)
  for name in op_names_to_remove:
      del ugraph.ops_info[name]
  if op_names_to_remove:
    topologic_order_graph(ugraph)
  return ugraph
