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

class _AttributesBase(object):

  def __init__(self):
    self.__attributes = {}
    for key in self._ALLOWED_KEYS.keys():
      if not isinstance(key, str):
        raise TypeError("allowed keys should be all string")

  def __setitem__(self, key, value):
    if not isinstance(key, str):
      raise TypeError("key must be string: {}".format(key))
    if key not in self._ALLOWED_KEYS:
      raise KeyError("{} is not allowed to added to {}".format(key, self))
    expect_type = self._ALLOWED_KEYS[key]
    if not isinstance(value, expect_type):
      raise TypeError(
        "expecting {} for {}, get {}".format(expect_type, key, type(value))
      )
    self.__attributes[key] = value

  def __getattr__(self, attrib):
    return getattr(self.__attributes, attrib)


def declare_attrib_cls(cls_name, allowed_keys=None):
  if allowed_keys is None:
    allowed_keys = {}
  else:
    allowed_keys = dict(allowed_keys)
  return type(cls_name, (_AttributesBase,), {"_ALLOWED_KEYS": allowed_keys})
