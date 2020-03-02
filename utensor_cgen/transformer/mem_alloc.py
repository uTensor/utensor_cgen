from collections import Counter, defaultdict, namedtuple
from itertools import chain, combinations, product

from ortools.sat.python import cp_model
from utensor_cgen.logger import logger
from utensor_cgen.utils import timed

from .base import Transformer
from .pipeline import TransformerPipeline

__all__ = ['TensorAllocationTransformer']

MemorySpan = namedtuple('MemorySpan', ['start', 'end', 'size'])
AllocationPlan = namedtuple('AllocationPlan', ['plan', 'total_size'])

@TransformerPipeline.register_transformer
class TensorAllocationTransformer(Transformer):
  METHOD_NAME = "tensor_alloc"
  KWARGS_NAMESCOPE = "_tensor_alloc"

  def __init__(
    self,
    prune_graph=False,
    max_pool_size=1024*1024, # 1k bytes
    vizualize_fname=None,
    **vizualize_kwargs
  ):
    super(TensorAllocationTransformer, self).__init__(prune_graph=prune_graph)
    self.max_pool_size = max_pool_size
    self.vizualize_fname = vizualize_fname
    self.vizualize_kwargs = vizualize_kwargs

  def transform(self, ugraph):
    ref_cnts = Counter()
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      for tensor in op_info.input_tensors:
        ref_cnts[tensor.name] += 1
    tensors_to_schedule = set()
    tensors_to_ignore = set(
      chain(*[
        op.output_tensors
        for op in ugraph.get_ops_by_type('Inline')
      ])
    )
    nonoverlap_map = defaultdict(set)
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      if op_info.op_type == 'Inline':
        continue
      # no overlapping between input/output tensors
      for in_tensor, out_tensor in product(op_info.input_tensors, op_info.output_tensors):
        if not in_tensor in tensors_to_ignore:
          nonoverlap_map[out_tensor].add(in_tensor)
          tensors_to_schedule.add(in_tensor)
      # no overlapping among input tensors
      for in_tensor1, in_tensor2 in combinations(
        filter(lambda tensor: tensor not in tensors_to_ignore, op_info.input_tensors),
        2
      ):
        nonoverlap_map[in_tensor1].add(in_tensor2)
      # no overlapping between output tensors and all evaluated tensors
      for known_tensor, out_tensor in product(tensors_to_schedule, op_info.output_tensors):
        if ref_cnts[known_tensor.name] > 0:
          nonoverlap_map[out_tensor].add(known_tensor)
      tensors_to_schedule.update(op_info.output_tensors)
      for in_tensor in op_info.input_tensors:
        ref_cnts[in_tensor.name] -= 1
      for out_tensor1, out_tensor2 in combinations(ugraph.output_tensors, 2):
        nonoverlap_map[out_tensor1].add(out_tensor2)
      tensors_to_schedule.update(ugraph.output_tensors)
    alloc_plan = self._solve_opt_plan(tensors_to_schedule, nonoverlap_map)
    if alloc_plan is not None:
      ugraph.attributes[self.KWARGS_NAMESCOPE] = alloc_plan
    if self.vizualize_fname:
      viz_memalloc(ugraph=ugraph, out_fname=self.vizualize_fname, **self.vizualize_kwargs)
    return ugraph

  @timed
  def _solve_opt_plan(self, tensors_to_schedule, nonoverlap_map):
    model = cp_model.CpModel()
    inter_vars = {}
    tensor_allocs = {}
    for tensor in tensors_to_schedule:
      var_start = model.NewIntVar(0, self.max_pool_size, '{}_start'.format(tensor.name))
      var_end = model.NewIntVar(0, self.max_pool_size, '{}_end'.format(tensor.name))
      size = tensor.size * tensor.dtype.itemsize
      intv_var = model.NewIntervalVar(var_start, size, var_end, '{}_alloc'.format(tensor.name))
      inter_vars[tensor.name] = intv_var
      tensor_allocs[tensor.name] = MemorySpan(var_start, var_end, size)
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
    alloc_plan = None
    if solver.StatusName(status) == 'OPTIMAL':
      opt_mempool_size = solver.Value(var_mempool_size)
      plan = {}
      for name, alloc in tensor_allocs.items():
        plan[name] = MemorySpan(
          start=solver.Value(alloc.start),
          end=solver.Value(alloc.end),
          size=alloc.size
        )
      logger.info('optimal tensor allocation plan solved, memory span: %i bytes', opt_mempool_size)
      alloc_plan = AllocationPlan(
        plan=plan,
        total_size=opt_mempool_size
      )
    else:
      logger.info('tensor allocation plan not found, status: %s', solver.StatusName(status))
    return alloc_plan

from utensor_cgen.ir.misc.graph_viz import viz_memalloc  # isort:skip
