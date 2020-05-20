import re
from itertools import chain
from pathlib import Path
from collections import defaultdict

from utensor_cgen.backend.base import BackendPart
from utensor_cgen.backend.utensor.snippets.composer import Composer
from utensor_cgen.backend.utensor.snippets.legacy import (
    ContextGlobalArrayContainer, WeightSnippet)
from utensor_cgen.backend.utensor.snippets.rearch import (
    DeclareRamTensorSnippet, DeclareRomTensorSnippet, 
    FreeTensorSnippet, SimpleContainer, TimeSlotContainer
)
from utensor_cgen.backend.utensor.snippets.template_env import env
from utensor_cgen.backend.graph_lower.generic_graph_lower import TopoOrderTensorTimeslotPlanner
from utensor_cgen.logger import logger
from utensor_cgen.utils import Configuration, class_property

from ._operators import OperatorFactory


class uTensorRearchCodeGenerator(BackendPart):

  TARGET = 'utensor'
  PART = 'rearch_code_generator'
  
  def __init__(self, config):
    final_config = Configuration(self.default_config, config)
    self.src_fname = final_config['src_fname']
    self.header_fname = final_config['header_fname']
    self.params_dir = final_config['params_dir'].rstrip('/')
    self.meta_data_pool_size = final_config['meta_data_pool_size']
    self.ram_data_pool_size = final_config['ram_data_pool_size']
    self.model_dir = final_config['model_dir'].rstrip('/')

  def apply(self, ugraph):
    # take a look of template file "simple.cpp", which is under templates/container/rearch/ directory
    # in submodule utensor_cgen.backend.utensor.snippets
    # you will see how declare_snippets and eval_snippets works and have better understanding of 
    # the rearch code generator
    src_fname = self.src_fname
    if src_fname == 'None':
      src_fname = '{}.cpp'.format(ugraph.name)
    # find all required ops and the variable names for the tensors in the generate files
    (
      ops,            # Set[Operator], no Placeholder or Inline ops
      placeholders,   # Set[String], variable names of planceholder tensors
      tensor_var_map, # dict, tensor name -> var name
    ) = self._find_required_ops(ugraph)
    is_success, weight_snippets, local_snippets = self._try_get_opt_snippets(ugraph, ops, tensor_var_map)
    # if user disable tensor life cycle analysis, such optimized snippets is not available
    if not is_success:
      # generate files
      self._naive_generate_files(
        ugraph,
        required_ops=ops,
        placeholders=placeholders,
        tensor_var_map=tensor_var_map
      )
    else:
      self._time_slot_generate_files(
        ugraph,
        placeholders=placeholders,
        tensor_var_map=tensor_var_map,
        weight_snippets=weight_snippets,
        local_snippets=local_snippets
      )

  def _find_required_ops(self, ugraph):
    # find all ops required
    ops = set()
    placeholders = set()
    tensor_var_map = {} # tensor name -> var name
    for op_info in ugraph.ops_info.values():
      for tensor in op_info.output_tensors:
        tensor_var_name = re.sub(r'[:/]', '', tensor.name)
        tensor_var_map[tensor.name] = tensor_var_name
        if op_info.op_type == 'Placeholder':
          placeholders.add(tensor_var_name)
      if op_info.op_type not in ['Placeholder', 'Inline']:
        ops.add(
          OperatorFactory.get_opertor(op_info)
        )
    return ops, placeholders, tensor_var_map

  def _try_get_opt_snippets(self, ugraph, ops, tensor_var_map):
    # snippets to be rendered in the generated function body
    local_snippets = []
    # snippets to be rendered in the weight header file
    weight_snippets = []
    # op -> op variable name
    ops_map = {}
    for i, op in enumerate(ops):
      op_var_name = 'op_{:03d}'.format(i)
      ops_map[op] = op_var_name
      # all ops will be declared first
      local_snippets.append(
        op.get_declare_snippet(op_var_name, tensor_var_map)
      )
    out_tensor_var_names = [
      tensor_var_map[tensor.name] for tensor in chain(*[
        ugraph.ops_info[op_name].output_tensors
        for op_name in ugraph.output_nodes
      ])
    ]

    # dict: tensor name -> TimeslotAllocation
    tensor_timeslots = ugraph.attributes.get(TopoOrderTensorTimeslotPlanner.KWARGS_NAMESCOPE)
    if tensor_timeslots is None:
      return False, weight_snippets, local_snippets
    time_slot_max = max(
      alloc.time_slot_start for alloc in tensor_timeslots.values()
    )
    time_slot_op_map = defaultdict(set)
    known_ops = set()
    for tensor_info, alloc in tensor_timeslots.items():
      op_info = tensor_info.op
      if op_info in known_ops:
        continue
      time_slot = alloc.time_slot_start
      time_slot_op_map[time_slot].add(
        op_info
      )
      known_ops.add(op_info)
    # populate local_snippets according to time slot analysis
    visited_tensors = set()
    freed_tensors = set()
    for current_time_slot in range(time_slot_max+1):
      for op_info in time_slot_op_map[current_time_slot]:
        if op_info.op_type == 'Placeholder':
          continue
        if op_info.op_type in ['Inline', 'Constant']:
          tensor_info = op_info.output_tensors[0]
          tensor_var = tensor_var_map[tensor_info.name]
          buffer_name = 'data_{}'.format(tensor_info.name.replace(':', '_').replace('/', '_'))
          weight_snippets.append(
            WeightSnippet(
              buffer_name,
              tensor_info.dtype,
              tensor_info.shape,
              op_info.op_attr['value'].value.np_array.ravel()
            )
          )
          local_snippets.append(
            DeclareRomTensorSnippet(
              tensor_info=tensor_info,
              tensor_var=tensor_var,
              buffer_var=buffer_name,
            )
          )
          visited_tensors.add(tensor_info)
        else:
          if op_info.name not in ugraph.output_nodes:
            for tensor_info in op_info.output_tensors:
              local_snippets.append(
                DeclareRamTensorSnippet(
                  tensor_info=tensor_info,
                  tensor_var=tensor_var_map[tensor_info.name]
                )
              )
              visited_tensors.add(tensor_info)
          op = OperatorFactory.get_opertor(op_info)
          op_name = ops_map[op]
          local_snippets.append(
            op.get_eval_snippet(
              op_name,
              op_info,
              tensor_var_map
            )
          )
      for tensor_info in visited_tensors.difference(freed_tensors):
        end_time_slot = tensor_timeslots[tensor_info].time_slot_end
        if end_time_slot is not None and current_time_slot >= end_time_slot:
          tensor_var = tensor_var_map[tensor_info.name]
          local_snippets.append(
            FreeTensorSnippet(tensor_var)
          )
          freed_tensors.add(tensor_info)
    return True, weight_snippets, local_snippets
  
  def _time_slot_generate_files(
    self, ugraph, placeholders, tensor_var_map, weight_snippets, local_snippets
  ):
    template_vars = {}
    template_vars['model_name'] = ugraph.name
    template_vars['meta_data_pool_size'] = self._compute_meta_data_size(ugraph)
    template_vars['ram_data_pool_size'] = self._compute_ram_data_size(ugraph)
    template_vars['placeholders'] = placeholders
    template_vars['out_tensor_var_names'] = [
      tensor_var_map[tensor.name] for tensor in chain(*[
        op_info.output_tensors for op_info in ugraph.output_ops
      ])
    ]
  
    params_dir = Path(self.params_dir) / ugraph.name
    params_dir.mkdir(parents=True, exist_ok=True)
    with (params_dir / 'params_{}.hpp'.format(ugraph.name)).open('w') as fid:
      fid.write("/* Auto-generated by utensor cli */\n")
      weight_container = ContextGlobalArrayContainer(
        snippets=weight_snippets
      )
      fid.write(weight_container.render())
      weight_header_fname = fid.name
      logger.info("model parameters header file generated: %s", fid.name)
    model_file_dir = Path(self.model_dir)
    header_fname = self.header_fname == 'None' and '{}.hpp'.format(ugraph.name) or self.header_fname
    (model_file_dir / ugraph.name).mkdir(parents=True, exist_ok=True)
    container_snippet = TimeSlotContainer()
    container_snippet.add_local_snippets(*local_snippets)
    container_snippet.template_vars.update(template_vars)
    container_snippet.add_header('"{}"'.format(weight_header_fname))
    with (model_file_dir / ugraph.name / header_fname).open('w') as fid:
      fid.write("/* Auto-generated by utensor cli */\n")
      template = env.get_template('snippets/rearch/simple.hpp')
      fid.write(template.render(**template_vars))
      container_snippet.add_header('"{}"'.format(fid.name))
      logger.info("model header file generated: %s", fid.name)

    composer = Composer(snippets=[container_snippet])
    src_fname = self.src_fname == 'None' and '{}.cpp'.format(ugraph.name) or self.src_fname
    with (model_file_dir / ugraph.name / src_fname ).open('w') as fid:
      fid.write("/* Auto-generated by utensor cli */\n")
      fid.write(composer.compose())
      logger.info("model cpp file generated: %s", fid.name)

  def _naive_generate_files(
    self, ugraph, required_ops, placeholders, tensor_var_map,
  ):
    (
        ops_map,                 # dict, op_info -> variable name of op in the output files
        out_tensor_var_names,    # list of tensor variable names, which are the variable names of the output tensors of the graph
        declare_global_snippets, # list of Snippet objects, which will rendered in global scop
        declare_local_snippets,  # list of Snippet objects, which will rendered in local function scope
        weight_snippets,         # snippets for generating weights header file
      ) = self._get_declare_snippets(ugraph, required_ops, tensor_var_map)
    # eval_snippets: List of snippet objects, which will render code snippets for tensor evaluation
    eval_snippets = self._get_evaluation_snippets(ugraph, ops_map, tensor_var_map)
    template_vars = {}
    template_vars['model_name'] = ugraph.name
    template_vars['meta_data_pool_size'] = self._compute_meta_data_size(ugraph)
    template_vars['ram_data_pool_size'] = self._compute_ram_data_size(ugraph)
    template_vars['placeholders'] = placeholders
    template_vars['out_tensor_var_names'] = out_tensor_var_names
    params_dir = Path(self.params_dir) / ugraph.name
    params_dir.mkdir(parents=True, exist_ok=True)
    weight_header_fname = None
    if weight_snippets:
      with (params_dir / 'params_{}.hpp'.format(ugraph.name)).open('w') as fid:
        fid.write("/* Auto-generated by utensor cli */\n")
        weight_container = ContextGlobalArrayContainer(
          snippets=weight_snippets
        )
        fid.write(weight_container.render())
        weight_header_fname = fid.name
        logger.info("model parameters header file generated: %s", fid.name)

    # generate the compute function
    model_file_dir = Path(self.model_dir)
    header_fname = self.header_fname == 'None' and '{}.hpp'.format(ugraph.name) or self.header_fname
    container_snippet = SimpleContainer()
    container_snippet.add_declare_global_snippets(*declare_global_snippets)
    container_snippet.add_declare_local_snippets(*declare_local_snippets)
    container_snippet.add_eval_snippets(*eval_snippets)
    container_snippet.template_vars.update(template_vars)
    (model_file_dir / ugraph.name).mkdir(parents=True, exist_ok=True)
    with (model_file_dir / ugraph.name / header_fname).open('w') as fid:
      fid.write("/* Auto-generated by utensor cli */\n")
      template = env.get_template('snippets/rearch/simple.hpp')
      fid.write(template.render(**template_vars))
      container_snippet.add_header(fid.name)
      logger.info("model header file generated: %s", fid.name)
    if weight_header_fname:
      container_snippet.add_header(weight_header_fname)
    composer = Composer(snippets=[container_snippet])
    src_fname = self.src_fname == 'None' and '{}.cpp'.format(ugraph.name) or self.src_fname
    with (model_file_dir / ugraph.name / src_fname ).open('w') as fid:
      fid.write("/* Auto-generated by utensor cli */\n")
      fid.write(composer.compose())
      logger.info("model cpp file generated: %s", fid.name)

  def _get_declare_snippets(self, ugraph, ops, tensor_var_map):
    # get ops/tensors declaration snippets
    declare_global_snippets = []
    declare_local_snippets = []
    ops_map = {} # op -> op variable name
    weight_snippets = []
    out_tensor_var_names = [
      tensor_var_map[tensor.name] for tensor in chain(*[
        ugraph.ops_info[op_name].output_tensors
        for op_name in ugraph.output_nodes
      ])
    ]
    for i, op in enumerate(ops):
      op_var_name = 'op_{:03d}'.format(i)
      ops_map[op] = op_var_name
      declare_local_snippets.append(op.get_declare_snippet(op_var_name, tensor_var_map))
    for op_info in filter(lambda op_info: op_info.op_type not in ["Inline", "Placeholder"], ugraph.ops_info.values()):
      if op_info.name in ugraph.output_nodes:
        continue
      for out_tensor in op_info.output_tensors:
        declare_local_snippets.append(
          DeclareRamTensorSnippet(out_tensor, tensor_var_map[out_tensor.name])
        )
    for op_info in filter(lambda op_info: op_info.op_type in ['Inline', 'Constant'], ugraph.ops_info.values()):
      tensor_info = op_info.output_tensors[0]
      tensor_var = tensor_var_map[tensor_info.name]
      buffer_name = 'data_{}'.format(tensor_info.name.replace(':', '_').replace('/', '_'))
      weight_snippets.append(
        WeightSnippet(
          buffer_name,
          tensor_info.dtype,
          tensor_info.shape,
          op_info.op_attr['value'].value.np_array.ravel()
        )
      )
      declare_local_snippets.append(
        DeclareRomTensorSnippet(
          tensor_info=tensor_info,
          tensor_var=tensor_var,
          buffer_var=buffer_name,
          # static=True,
        )
      )
    return ops_map, out_tensor_var_names, declare_global_snippets, declare_local_snippets, weight_snippets

  def _get_evaluation_snippets(self, ugraph, ops_map, tensor_var_map):
    eval_snippets = []
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      if op_info.op_type in ['Placeholder', 'Inline']:
        continue
      op = OperatorFactory.get_opertor(op_info)
      op_name = ops_map[op]
      eval_snippets.append(
        op.get_eval_snippet(op_name, op_info, tensor_var_map)
      )
    return eval_snippets


  @class_property
  def default_config(cls):
    config = {}
    config['src_fname'] = 'None'
    config['header_fname'] = 'None'
    config['params_dir'] = 'constants'
    config['model_dir'] = 'models'
    config['meta_data_pool_size'] = 'auto'
    config['ram_data_pool_size'] = 'auto'
    return config

  def _compute_meta_data_size(self, ugraph):
    # TODO: if mem_optimizer is None, use a default mem optimizer
    if self.meta_data_pool_size == 'auto':
      # TODO: compute actual meta data size with ugraph
      size = 2048
    else:
      size = self.meta_data_pool_size
    return size

  def _compute_ram_data_size(self, ugraph):
    # TODO: if mem_optimizer is None, use a default mem optimizer
    if self.ram_data_pool_size == 'auto':
      # TODO: compute actual ram data size with ugraph
      if '_tensor_alloc' in ugraph.attributes:
        size = ugraph.attributes['_tensor_alloc'].total_size + 3000
      else:
        size = 256
    else:
      size = self.ram_data_pool_size
    return size
