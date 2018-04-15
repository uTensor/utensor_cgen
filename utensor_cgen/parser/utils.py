# -*- coding: utf8 -*-
from collections import defaultdict
from copy import deepcopy

def clusters_by_name_scopes(op_infos, name_scope_prefix=None):
  """
  Arguements
  ----------
  op_infos : list[OperationInfo]
      list of parser.OperationInfo
  name_scope_prefix : str
      the target name scope prefix, e.g `dropout`.

  Return
  ------
  clusters : dict
      a dictionary of found name_scopes as key and list of
      operation names as value
  """
  op_infos_copy = deepcopy(op_infos)
  if name_scope_prefix is not None:
    op_infos_copy = [op_info for op_info in op_infos_copy
                     if op_info.node_name.startswith(name_scope_prefix)]
  current_name_scope = op_infos[0].node_name.split('/')[0]
  name_scope_map = defaultdict(lambda: [])

  visited = set([])
  for op_info in op_infos_copy:
    cluster = []
    queue = []
    if op_info.node_name in visited:
      continue
