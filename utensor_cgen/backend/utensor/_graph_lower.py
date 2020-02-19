from utensor_cgen.backend.base import BackendPart
from utensor_cgen.utils import class_property
from .._types import NP_TYPES_MAP

class uTensorGraphLower(BackendPart):

  TARGET = 'utensor'
  PART = 'graph_lower'

  def apply(self, ugraph):
    handler = getattr(self, 'handle_{}'.format(ugraph.lib_name))
    if handler is None:
      raise RuntimeError(
        'can not lower ugraph from {} to utensor'.format(ugraph.lib_name)
      )
    return handler(ugraph)

  def handle_tensorflow(self, ugraph):
    return ugraph

  @class_property
  def default_config(cls):
    return {}
 

 class uTensorOfflineMemoryPlanner(BackendPart):

   TARGET = 'utensor'
   PART = 'offlinememory'


  def __init__(
    self,
    buff_size=100000, #1k bytes
    unit_size=4,
    config
  ):
    self.buff_size = buff_size
    self.unit_size = unit_size
    final_config = Configuration(self.default_config, config)
    self._type = {}
    self._type[np.dtype(tf.float32.as_numpy_dtype)] = final_config['size_float']
    self._type[np.dtype(tf.int32.as_numpy_dtype)] = final_config['size_int']
    self._type[np.dtype(tf.quint8.as_numpy_dtype)] = final_config['size_uint8_t']
    self._type[np.dtype(tf.qint8.as_numpy_dtype)] = final_config['size_uint8_t']
    self._type[np.dtype(tf.qint32.as_numpy_dtype)] = final_config['size_int']

   def apply(self, ugraph):
    new_ugraph = deepcopy(ugraph)
    new_ugraph.setup_data_manager({self.DATA_NAME: " "})
    # use_def_table: dict, tensor_name -> {'start': op_idx, 'end': op_idx}
    use_def_table = self._create_resource_table(new_ugraph)
    allocate_table = dict()
    allocate_success = self.allocate_graph(new_ugraph, allocate_table, use_def_table, self.buff_size, self.unit_size)

    if allocate_success:
      for node_name in new_ugraph.topo_order:
        in_t_infos = new_ugraph.ops_info[node_name].input_tensors
        for in_o in in_t_infos:
          if in_o.name in allocate_table:
            new_ugraph.data_manager.address = (in_o.name, allocate_table[in_o.name]['offsetstart'])
        out_t_infos = new_ugraph.ops_info[node_name].output_tensors
        for out_o in out_t_infos:
          if out_o.name in allocate_table:
            new_ugraph.data_manager.address = (out_o.name, allocate_table[out_o.name]['offsetstart'])
      return new_ugraph
    return ugraph

  def _query_offset_fromallocate_table(self, allocate_table, start, end):
    new_start = start
    new_end = end
    for key in allocate_table:
      if allocate_table[key]['offsetstart'] >= start and allocate_table[key]['offsetend'] <= end:
        continue
      elif allocate_table[key]['offsetstart'] <= start and allocate_table[key]['offsetend'] >= start:
        new_start = allocate_table[key]['offsetstart']
        if allocate_table[key]['offsetend'] >= end:
          new_end = max(new_end, allocate_table[key]['offsetend'])
        else:
          new_end = max(end, new_end)
      elif allocate_table[key]['offsetstart'] >= start and allocate_table[key]['offsetend'] >= start:
        if allocate_table[key]['offsetend'] >= end:
          new_end = max(new_end, allocate_table[key]['offsetend'])
        else:
          new_end = max(end, new_end)
    return new_start, new_end

  def _query_time_fromallocate_table(self, allocate_table, start, end):
    time_start = start
    time_end = end
    for key in allocate_table:
      if allocate_table[key]['start'] >= start and allocate_table[key]['end'] <= end:
        continue
      elif allocate_table[key]['start'] <= start and allocate_table[key]['end'] >= start:
        if allocate_table[key]['end'] >= end:
          time_end = max(time_end, allocate_table[key]['end'])
        else:
          time_end = max(end, time_end)
      elif allocate_table[key]['start'] >= start and allocate_table[key]['end'] >= start:
        if allocate_table[key]['end'] >= end:
          time_end = max(time_end, allocate_table[key]['end'])
        else:
          time_end = max(end, time_end)
    return time_start, time_end

  def _query_result(self, allocate_table, offset, length, timestart, timeend):
    for key in allocate_table:
      mem_occupied = (
        (allocate_table[key]['offsetstart'] >= offset and allocate_table[key]['offsetstart'] <= offset + length) or
        (allocate_table[key]['offsetstart'] <= offset and allocate_table[key]['offsetend'] >= offset)
      )
      life_span_occupied = (
        (allocate_table[key]['start'] >= timestart and allocate_table[key]['start'] <= timeend) or
        (allocate_table[key]['start'] <= timestart and allocate_table[key]['end'] >= timestart)
      )
      if mem_occupied and life_span_occupied:
        return True
    return False
  
  def allocate_tensor(self, tensors, tensor_index, allocate_table, use_def_table, buffer_size, unit_size):
    if tensor_index == len(tensors):
      return True
    if tensors[tensor_index].name in allocate_table:
      return self.allocate_tensor(tensors, tensor_index + 1, allocate_table, use_def_table, buffer_size, unit_size)

    tensor = tensors[tensor_index]
    candidates = self._get_candidates(allocate_table, use_def_table, buffer_size, unit_size, tensor)
    if not candidates:
      return False
    success = False
    for candidate in candidates:
      self._update_allocation_table(allocate_table, use_def_table, tensor, candidate, candidate + tensor.size)
      success = self.allocate_tensor(tensors, tensor_index + 1, allocate_table, use_def_table, buffer_size, unit_size)
      if success:
        break
      else:
        self._remove_allocate_table(allocate_table, tensor)
    return success

  def allocate_graph(self, ugraph, allocate_table, use_def_table, buffer_size, unit_size):
    tensors = []

    for node_name in ugraph.topo_order:
      in_t_infos = [
        tensor
        for tensor in ugraph.ops_info[node_name].input_tensors
        if tensor.op.op_type != 'Inline'
      ]
      out_t_infos = [
        tensor
        for tensor in ugraph.ops_info[node_name].output_tensors
        if tensor.op.op_type != 'Inline'
      ]
      tensors.extend(in_t_infos)
      tensors.extend(out_t_infos)
    
    succ = self.allocate_tensor(tensors, 0, allocate_table, use_def_table, buffer_size, unit_size)
    return succ

  def _check(self, allocate_table, use_def_table, tensor, tensor_offset_start, tensor_offset_end):
    valid = False
    timestart = use_def_table[tensor.name]['start']
    timeend = use_def_table[tensor.name]['end']
    offset, length = self._query_offset_fromallocate_table(allocate_table, tensor_offset_start, tensor_offset_end)
    timestart, timeend = self._query_time_fromallocate_table(allocate_table, timestart, timeend)
    occupied = self._query_result(allocate_table, offset, length, timestart, timeend)
    if not occupied:
      valid = True
    return valid

  def _get_candidates(self, allocate_table, use_def_table, buffer_size, unit_size, in_o):
    ret = []
    for i in range(0, buffer_size, unit_size):
      if self._check(allocate_table, use_def_table, in_o, i, i + in_o.size * self._type[in_o.dtype]):
        ret.append(i)
    return ret
  
  def _update_allocation_table(
    self,
    allocate_table,
    use_def_table,
    tensor,
    offset_start,
    offset_end
  ):   
    time_start = use_def_table[tensor.name]['start']
    time_end = use_def_table[tensor.name]['end']
    attribute = dict()
    attribute['start'] = time_start
    attribute['end'] = time_end
    attribute['offsetstart'] = offset_start
    attribute['offsetend'] = offset_end
    allocate_table[tensor.name] = attribute
    return allocate_table   
  
  def _remove_allocate_table(self, allocate_table, tensor):
    del allocate_table[tensor.name]

  def _create_resource_table(self, ugraph):
    resource_table = dict()
    len_map = {
      op_name: idx
      for idx, op_name in enumerate(ugraph.topo_order)
    }
    for node_name in ugraph.topo_order:
      for tensor_info in ugraph.ops_info[node_name].input_tensors:
        if tensor_info.name not in resource_table:
          lifetime = dict()
          lifetime['start'] = len_map[node_name]
          lifetime['end'] = len_map[node_name]  
          resource_table[tensor_info.name] = lifetime   
        resource_table[tensor_info.name]['end']= len_map[node_name] 
      
      for outtensor in ugraph.ops_info[node_name].output_tensors:
        if outtensor.name not in resource_table:
          lifetime = dict()
          lifetime['start'] = len_map[node_name]
          lifetime['end'] = len_map[node_name]  
          resource_table[outtensor.name] = lifetime

    return resource_table


    @class_property
    def default_config(cls):
      config = {}
      config['size_float'] = 4
      config['size_int'] = 4
      config['size_uint8_t'] = 1
      
      return config