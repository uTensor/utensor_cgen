# -*- coding: utf8 -*-
def is_list_of(vtype):
  """
  validator for ``attr``

  Check if the given value is a list of specific type
  """
  def check(inst, attrib, value):
    is_valid = True
    if not isinstance(value, list):
      is_valid = False
    else:
      for v in value:
        if not isinstance(v, vtype):
          is_valid = False
    if not is_valid:
      raise TypeError('Expecting list of type %s, get %s' % (vtype, value))
  return check

def check_integrity(ugraph):
  """Deprecated
  """
  for op_name, op_info in ugraph.ops_info.items():
    for input_tensor_info in op_info.input_tensors:
      assert input_tensor_info.op_name in ugraph.ops_info, \
        (
          "In %r: input tensor %r points to non-existing op %r" 
          % (op_name, input_tensor_info.name, input_tensor_info.op_name)
        )
      assert input_tensor_info.op_name in ugraph.topo_order, \
        (
          "In %r: input tensor %r points to an op (%r) that does not exist in graph.topo_order"
          % (op_name, input_tensor_info.name, input_tensor_info.op_name)
        )
  assert len(ugraph.ops_info) == len(ugraph.topo_order)
