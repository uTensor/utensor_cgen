from collections import Counter

def get_refc_table(graph_info) -> Counter:
  refc_table = Counter()
  for op_name in graph_info:
    op_info = graph_info[op_name]

    if op_info["op_type"] in ["Const"]:
      refc_table[op_name] += 1
    else:
      for tName, _, _ in op_info["input_tensor"]:
        refc_table[tName] += 1
      #add to ref table

  return refc_table

#def get_tensor_size(tensor_name, )
