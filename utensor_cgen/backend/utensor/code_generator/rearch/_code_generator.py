import os
import pickle
import re
from itertools import chain
from pathlib import Path

from utensor_cgen.backend.base import BackendPart
from utensor_cgen.backend.utensor.snippets.composer import Composer
from utensor_cgen.backend.utensor.snippets.legacy import (
    ContextGlobalArrayContainer, WeightSnippet)
from utensor_cgen.backend.utensor.snippets.rearch import SimpleContainer
from utensor_cgen.backend.utensor.snippets.template_env import env
from utensor_cgen.transformer.pipeline import TransformerPipeline
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
    self.trans_methods = final_config['transform_methods']
    self.meta_data_pool_size = final_config['meta_data_pool_size']
    self.ram_data_pool_size = final_config['ram_data_pool_size']
    self.model_dir = final_config['model_dir'].rstrip('/')
    self.save_graph = final_config['save_graph']

  def apply(self, ugraph):
    src_fname = self.src_fname
    if src_fname == 'None':
      src_fname = '{}.cpp'.format(ugraph.name)
    pipeline = TransformerPipeline(self.trans_methods)
    new_ugraph = pipeline.transform(ugraph)
    if self.save_graph:
      with open('transformed_{}.pkl'.format(ugraph.name), 'wb') as fid:
        pickle.dump(new_ugraph, fid)
    # 1. find all ops required
    ops = set()
    placeholders = set()
    tensor_var_map = {} # tensor name -> var name
    for op_info in new_ugraph.ops_info.values():
      for tensor in op_info.output_tensors:
        tensor_var_name = re.sub(r'[:/]', '', tensor.name)
        tensor_var_map[tensor.name] = tensor_var_name
        if op_info.op_type == 'Placeholder':
          placeholders.add(tensor_var_name)
      if op_info.op_type not in ['Placeholder', 'Inline']:
        ops.add(
          OperatorFactory.get_opertor(op_info)
        )
    # 2. ops/tensors declaration
    declare_snippets = []
    ops_map = {} # op -> op variable name
    for i, op in enumerate(ops):
      op_var_name = 'op_{:03d}'.format(i)
      ops_map[op] = op_var_name
      declare_snippets.append(op.get_declare_snippet(op_var_name))
    weight_snippets = []
    for op_info in filter(lambda op_info: op_info.op_type == 'Inline', new_ugraph.ops_info.values()):
      tensor = op_info.output_tensors[0]
      buffer_name = 'data_{}'.format(tensor.name.replace(':', '_').replace('/', '_'))
      weight_snippets.append(
        WeightSnippet(
          buffer_name,
          tensor.dtype,
          tensor.shape,
          op_info.op_attr['value'].value.np_array.ravel()
        )
      )
      declare_snippets.append(
        OperatorFactory.get_opertor(op_info).get_declare_snippet(
          tensor_var_name=tensor_var_map[tensor.name],
          buffer_var_name=buffer_name,
          tensor=tensor
        )
      )
    # 3. evaluation snippets
    eval_snippets = []
    for op_name in new_ugraph.topo_order:
      op_info = new_ugraph.ops_info[op_name]
      if op_info.op_type in ['Placeholder', 'Inline']:
        continue
      op = OperatorFactory.get_opertor(op_info)
      op_name = ops_map[op]
      eval_snippets.append(
        op.get_eval_snippet(op_info, op_name, tensor_var_map)
      )
    template_vars = {}
    template_vars['model_name'] = ugraph.name
    template_vars['meta_data_pool_size'] = self._compute_meta_data_size(new_ugraph)
    template_vars['ram_data_pool_size'] = self._compute_ram_data_size(new_ugraph)
    template_vars['placeholders'] = placeholders
    template_vars['out_tensor_var_names'] = [
      tensor_var_map[tensor.name] for tensor in chain(*[
        new_ugraph.ops_info[op_name].output_tensors
        for op_name in new_ugraph.output_nodes
      ])
    ]
    # 4. write files
    params_dir = Path(self.params_dir) / ugraph.name
    params_dir.mkdir(parents=True, exist_ok=True)
    weight_header_fname = None
    if weight_snippets:
      with (params_dir / 'params_{}.hpp'.format(ugraph.name)).open('w') as fid:
        weight_container = ContextGlobalArrayContainer(snippets=weight_snippets)
        fid.write(weight_container.render())
        weight_header_fname = fid.name

    # # generate the computation function
    model_file_dir = Path(self.model_dir)
    header_fname = self.header_fname == 'None' and '{}.hpp'.format(ugraph.name) or self.header_fname
    container_snippet = SimpleContainer(declare_snippets=declare_snippets, eval_snippests=eval_snippets)
    container_snippet.template_vars.update(template_vars)
    (model_file_dir / ugraph.name).mkdir(parents=True, exist_ok=True)
    with (model_file_dir / ugraph.name / header_fname).open('w') as fid:
      template = env.get_template('snippets/rearch/simple.hpp')
      fid.write(template.render(**template_vars))
      container_snippet.add_header(fid.name)
    if weight_header_fname:
      container_snippet.add_header(weight_header_fname)
    composer = Composer(snippets=[container_snippet])
    src_fname = self.src_fname == 'None' and '{}.cpp'.format(ugraph.name) or self.src_fname
    with (model_file_dir / ugraph.name / src_fname ).open('w') as fid:
      fid.write(composer.compose())

  @class_property
  def default_config(cls):
    config = {}
    config['src_fname'] = 'None'
    config['header_fname'] = 'None'
    config['params_dir'] = 'data'
    config['model_dir'] = 'models'
    config['transform_methods'] = [
      'dropout(name_pattern=r"(dropout[_\w\d]*)/.*")',
      # 'linear_reorder',
      # 'quantize',
      # 'conv_pool',
      'inline',
      'biasAdd',
      'remove_id_op',
      'fake_gather_v2',
      # 'refcnt'
    ]
    config['meta_data_pool_size'] = 'auto'
    config['ram_data_pool_size'] = 'auto'
    config['save_graph'] = False
    return config

  def _compute_meta_data_size(self, ugraph):
    if self.meta_data_pool_size == 'auto':
      # TODO: compute actual meta data size with ugraph
      size = 256
    else:
      size = self.meta_data_pool_size
    return size

  def _compute_ram_data_size(self, ugraph):
    if self.ram_data_pool_size == 'auto':
      # TODO: compute actual ram data size with ugraph
      size = 256
    else:
      size = self.ram_data_pool_size
    return size
