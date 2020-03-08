r"""Generic Graph Lowering

All graph lower in this module should be generic.
That is, it should be able to apply to any graph (i.e target independent)
"""
import os
import pickle
from collections import Counter, defaultdict, namedtuple
from copy import deepcopy
from itertools import chain, combinations, product
from math import ceil, log10

import numpy as np
import tensorflow as tf
from ortools.sat.python import cp_model
from utensor_cgen.backend.base import BackendPart
from utensor_cgen.backend.utensor.snippets._types import NP_TYPES_MAP
from utensor_cgen.logger import logger
from utensor_cgen.utils import Configuration, class_property, timed

from .alloc_plan import (AllocationPlan, SpaceAllocation, TimeslotAllocation,
                         TimeSpaceAllocation)

__all__ = ['TensorAllocationPlanner', 'BrutalForceMemoryPlanner']

_VarMemorySpan = namedtuple('_VarMemorySpan', ['start', 'end', 'size'])

class TopoOrderTensorTimeslotPlanner(BackendPart):
  TARGET = 'generic'
  PART = 'tensor_timeslot_planner'
  KWARGS_NAMESCOPE = '_tensor_timeslot'

  @timed
  def apply(self, ugraph):
    ref_cnts = Counter()
    life_span = defaultdict(lambda: [None, None])
    for op_info in ugraph.ops_info.values():
      for tensor in op_info.input_tensors:
        ref_cnts[tensor.name] += 1
    for time_slot, op_name in enumerate(ugraph.topo_order):
      op_info = ugraph.ops_info[op_name]
      for tensor in op_info.output_tensors:
        life_span[tensor.name][0] = time_slot
      for tensor in op_info.input_tensors:
        ref_cnts[tensor.name] -= 1
        if ref_cnts[tensor.name] == 0:
          life_span[tensor.name][1] = time_slot
    time_alloc_plan = {}
    for tensor_name, (start, end) in life_span.items():
      time_alloc = TimeslotAllocation(
        time_slot_start=start,
        time_slot_end=end
      )
      time_alloc_plan[tensor_name] = time_alloc
    logger.info('topo ordered tensor life span analysis done')
    ugraph.attributes[self.KWARGS_NAMESCOPE] = time_alloc_plan

  @class_property
  def default_config(cls):
    return {}


class TensorAllocationPlanner(BackendPart):
  """
  Offline Tensor Allocation Optimizer

  analyse tensor lifetime and find the optimal allocation offset in the managed memory pool

  :param max_pool_size: the size of the memory pool (default: 1 KB)
  :type max_pool_size: int

  :param include_inputs: include the input tensors (Placeholder) in the allocation plan
  :type include_inputs: bool

  :param out_fname: the file name of memory allocation visualization (will NOT generate if not given)
  :type out_fname: str

  :param aesthetic_kwargs: the keyword arguments controlling the aesthetic of the visualization of allocation plan
  :type aesthetic_kwargs: dict
  """
  TARGET = 'generic'
  PART = 'tensor_alloc_planner'
  KWARGS_NAMESCOPE = "_tensor_alloc"

  def __init__(self, config):
    self.max_pool_size = self.config['max_pool_size']
    self.include_inputs = self.config['include_inputs']
    self.out_fname = self.config['out_fname']
    if self.out_fname == 'None':
      self.out_fname = None
    aesthetic_kwargs = self.config['aesthetic_kwargs']
    if aesthetic_kwargs['figsize'] == 'None':
      aesthetic_kwargs['figsize'] = None
    self.aesthetic_kwargs = aesthetic_kwargs.to_dict()
    self.enabled = self.config['enabled']
    self.dtype_size_map = self._parse_dtype_size_map(self.config)

  def apply(self, ugraph):
    if not self.enabled:
      # not enabled, do nothing
      return
    time_alloc_plan = ugraph.attributes.get(
      TopoOrderTensorTimeslotPlanner.KWARGS_NAMESCOPE
    )
    if time_alloc_plan is None:
      TopoOrderTensorTimeslotPlanner(config={}).apply(ugraph)
      time_alloc_plan = ugraph.attributes[TopoOrderTensorTimeslotPlanner.KWARGS_NAMESCOPE]
    tensors_to_ignore = set(
      chain(*[
        op.output_tensors
        for op in ugraph.get_ops_by_type('Inline')
      ])
    )
    if not self.include_inputs:
      tensors_to_ignore.update(
        chain(*[
          op.output_tensors
          for op in ugraph.get_ops_by_type('Placeholder')
        ])
      )
    tensors_to_schedule = set()
    nonoverlap_map = defaultdict(set)
    for time_slot, op_name in enumerate(ugraph.topo_order):
      op_info = ugraph.ops_info[op_name]
      # ignore inline ops
      if op_info.op_type == 'Inline':
        continue
      # ignore placeholders if not included
      if not self.include_inputs and op_info.op_type == 'Placeholder':
        continue
      # all output tensor should not overlap with tensors that's still alive
      for out_tensor, known_tensor in product(op_info.output_tensors, tensors_to_schedule):
        time_alloc = time_alloc_plan[known_tensor.name]
        if time_slot in time_alloc:
          nonoverlap_map[out_tensor].add(known_tensor)
      # all output tensors should not overlap with each other
      for out_tensor1, out_tensor2 in combinations(op_info.output_tensors, 2):
        nonoverlap_map[out_tensor1].add(out_tensor2)
      # update tensors to be scheduled
      tensors_to_schedule.update(op_info.output_tensors)
    space_alloc_plan, opt_mempool_size = self._solve_space_alloc(tensors_to_schedule, nonoverlap_map)
    time_space_allocs = []
    if space_alloc_plan:
      for tensor_name in space_alloc_plan:
        space_alloc = space_alloc_plan[tensor_name]
        time_alloc = time_alloc_plan[tensor_name]
        time_space_allocs.append(
            TimeSpaceAllocation(
              tensor_name,
              time_alloc=time_alloc,
              space_alloc=space_alloc
          )
        )
      ugraph.attributes[self.KWARGS_NAMESCOPE] = AllocationPlan(
        allocs=time_space_allocs,
        total_size=opt_mempool_size
      )
      if self.out_fname:
        figs = viz_memalloc(ugraph=ugraph, **self.aesthetic_kwargs)
        if len(figs) == 1:
          logger.info('saving tensor mem allocation to %s', self.out_fname)
          figs[0].savefig(self.out_fname)
        else:
          num_digits = ceil(log10(len(figs)))
          file_format = '{{}}_{{:0{}d}}{{}}'.format(num_digits)
          for i, fig in enumerate(figs, 1):
            fname, ext = os.path.splitext(self.out_fname)
            fname = file_format.format(fname, i, ext)
            logger.info('saving tensor mem allocation to %s', fname)
            fig.savefig(fname)
        with open('{}.pkl'.format(self.out_fname), 'wb') as fid:
          pickle.dump(figs, fid)
          logger.info('matplotlib figure object dumped (pickle): %s', fid.name)

  @timed
  def _solve_space_alloc(self, tensors_to_schedule, nonoverlap_map):
    model = cp_model.CpModel()
    inter_vars = {}
    tensor_allocs = {}
    for tensor in tensors_to_schedule:
      var_start = model.NewIntVar(0, self.max_pool_size, '{}_start'.format(tensor.name))
      var_end = model.NewIntVar(0, self.max_pool_size, '{}_end'.format(tensor.name))
      size = self._compute_tensor_bytes_size(tensor)
      intv_var = model.NewIntervalVar(var_start, size, var_end, '{}_alloc'.format(tensor.name))
      inter_vars[tensor.name] = intv_var
      tensor_allocs[tensor.name] = _VarMemorySpan(var_start, var_end, size)
    for tensor in tensors_to_schedule:
      inter_var = inter_vars[tensor.name]
      nonoverlap_vars = [inter_vars[t.name] for t in nonoverlap_map[tensor]]
      for other in nonoverlap_vars:
          model.AddNoOverlap([inter_var, other])
    var_mempool_size = model.NewIntVar(0, self.max_pool_size, 'mempool_size')
    model.AddMaxEquality(var_mempool_size, [alloc.end for alloc in tensor_allocs.values()])
    model.Minimize(var_mempool_size)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    alloc_plan = {}
    opt_mempool_size = None
    if solver.StatusName(status) == 'OPTIMAL':
      opt_mempool_size = solver.Value(var_mempool_size)
      for name, alloc in tensor_allocs.items():
        alloc_plan[name] = SpaceAllocation(
          offset_start=solver.Value(alloc.start),
          size=alloc.size
        )
      logger.info('optimal tensor allocation plan solved, memory span: %i bytes', opt_mempool_size)
    else:
      logger.info('tensor allocation plan not found, status: %s', solver.StatusName(status))
    return alloc_plan, opt_mempool_size

  def _compute_tensor_bytes_size(self, tensor_info):
    size = tensor_info.size
    elem_size = self.dtype_size_map.get(tensor_info.dtype, tensor_info.dtype.itemsize)
    return elem_size * size

  @classmethod
  def _parse_dtype_size_map(cls, config):
    dtype_size_map = {}
    for dtype_str, size in config['dtype_size_map'].items():
      if dtype_str == 'float':
        dtype_size_map[np.dtype('float32')] = size
      elif dtype_str == 'uint8_t':
        dtype_size_map[np.dtype('uint8')] = size
      else:
        # TODO: user-defined data types
        # https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
        dtype_size_map[np.dtype(dtype_str)] = size
    return dtype_size_map

  @class_property
  def default_config(cls):
    return {
      'max_pool_size': 1024*1024, # 1k bytes
      'include_inputs': False,
      'out_fname': 'None',
      'aesthetic_kwargs': {
        'split_on_large_graph': True,
        'num_tensors_per_split': 20,
        'figsize': 'None',
        'fontsize': 12,
        'lw': 12,
        'rand_seed': 1111
      },
      'enabled': True,
      'dtype_size_map': {
        'float': 4,
        'double': 8,
        'uint8': 1,
        'int': 4,
        'long': 8,
      },
    }


class BrutalForceMemoryPlanner(BackendPart):
  TARGET = 'utensor'
  PART = 'offlinememory'
  DATA_NAME = 'address'

  def __init__(
    self,
    config,
    buff_size=100000, #1k bytes
    unit_size=4,
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

# FIXME: cyclic import
from utensor_cgen.ir.misc.graph_viz import viz_memalloc  # isort:skip
