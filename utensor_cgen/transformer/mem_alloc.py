from collections import Counter, defaultdict, namedtuple
from itertools import combinations, product

from ortools.sat.python import cp_model

from .base import Transformer
from .pipeline import TransformerPipeline

__all__ = ['TensorAllocationTransformer']

Allocation = namedtuple('Allocation', ['start', 'end', 'size'])
AllocationPlan = namedtuple('AllocationPlan', ['plan', 'total_size'])

@TransformerPipeline.register_transformer
class TensorAllocationTransformer(Transformer):
    METHOD_NAME = "tensor_alloc"
    KWARGS_NAMESCOPE = "_tensor_alloc"

    def __init__(self, prune_graph=True, max_pool_size=1024*1024*1024):
      super(TensorAllocationTransformer, self).__init__(prune_graph=prune_graph)
      self.max_pool_size = max_pool_size

    def transform(self, ugraph):
      ref_cnts = Counter()
      for op_name in ugraph.topo_order:
        op_info = ugraph.ops_info[op_name]
        for tensor in op_info.input_tensors:
          ref_cnts[tensor.name] += 1
      visited_tensors = set()
      nonoverlap_map = defaultdict(set)
      for op_name in ugraph.topo_order:
        op_info = ugraph.ops_info[op_name]
        for in_tensor, out_tensor in product(op_info.input_tensors, op_info.output_tensors):
          nonoverlap_map[out_tensor].add(in_tensor)
          visited_tensors.add(in_tensor)
      for known_tensor, out_tensor in product(visited_tensors, op_info.output_tensors):
        if ref_cnts[known_tensor.name] > 0:
          nonoverlap_map[out_tensor].add(known_tensor)
      for in_tensor in op_info.input_tensors:
        ref_cnts[in_tensor.name] -= 1
      for out_tensor1, out_tensor2 in combinations(ugraph.output_tensors, 2):
        nonoverlap_map[out_tensor1].add(out_tensor2)
      visited_tensors.update(ugraph.output_tensors)

      model = cp_model.CpModel()
      inter_vars = {}
      tensor_allocs = {}
      for tensor in visited_tensors:
        var_start = model.NewIntVar(0, self.max_pool_size, f'{tensor.name}_start')
        var_end = model.NewIntVar(0, self.max_pool_size, f'{tensor.name}_end')
        size = tensor.size * tensor.dtype.itemsize
        intv_var = model.NewIntervalVar(var_start, size, var_end, f'{tensor.name}_alloc')
        inter_vars[tensor.name] = intv_var
        tensor_allocs[tensor.name] = Allocation(var_start, var_end, size)
      for tensor in visited_tensors:
        inter_var = inter_vars[tensor.name]
        nonoverlap_vars = [inter_vars[t.name] for t in nonoverlap_map[tensor]]
        for other in nonoverlap_vars:
            model.AddNoOverlap([inter_var, other])
      var_mem_span = model.NewIntVar(0, self.max_pool_size, f'mem_span')
      model.AddMaxEquality(var_mem_span, [alloc.end for alloc in tensor_allocs.values()])
      model.Minimize(var_mem_span)

      solver = cp_model.CpSolver()
      status = solver.Solve(model)
      if solver.StatusName(status) == 'OPTIMAL':
        opt_mem_span = solver.Value(var_mem_span)
        alloc_plan = {}
        for name, alloc in tensor_allocs.items():
          alloc_plan[name] = Allocation(
            start=solver.Value(alloc.start),
            end=solver.Value(alloc.end),
            size=alloc.size
          )
        ugraph.attributes[self.KWARGS_NAMESCOPE] = AllocationPlan(
          plan=alloc_plan,
          total_size=opt_mem_span
        )
      return ugraph
