import os

from utensor_cgen.backend.base import BackendPart
from utensor_cgen.transformer.pipeline import TransformerPipeline
from utensor_cgen.utils import Configuration, class_property
from utensor_cgen.backend.utensor.snippets import env

class uTensorRearchCodeGenerator(BackendPart):

  TARGET = 'utensor'
  PART = 'rearch_code_generator' 
  
  def __init__(self, config):
    final_config = Configuration(self.default_config, config)
    self.src_fname = final_config['src_fname']
    self.params_dir = final_config['params_dir'].rstrip('/')
    self.trans_methods = final_config['transform_methods']
    if not os.path.exists(self.params_dir):
      os.makedirs(self.params_dir)
    self.meta_data_pool_size = final_config['meta_data_pool_size']
    self.ram_data_pool_size = final_config['ram_data_pool_size']
    self.embed_data_dir = final_config['embed_params_dir'].rstrip('/')
    self.model_dir = final_config['model_dir'].rstrip('/')
    self.save_graph = final_config['save_graph']
    self.debug_cmt = final_config['debug_cmt']

  def apply(self, ugraph):
    src_fname = self.src_fname
    if src_fname == 'None':
      src_fname = '{}.cpp'.format(ugraph.name)
    pipeline = TransformerPipeline(self.trans_methods)
    new_ugraph = pipeline.transform(ugraph)
    # 1. find all ops required
    required_op_types = set()
    placeholders = set()
    placeholders_var_map = {}
    for op_info in new_ugraph.ops_info.values():
      if op_info.op_type == 'Placeholder':
        for tensor in op_info.output_tensors:
          var_name = tensor.name.replace(':', '_')
          placeholders.add(var_name)
          placeholders_var_map[tensor.name] = var_name
      elif op_info.op_type != 'Inline':
        required_op_types.add(op_info.op_type)
    ops_map = {}
    for i, op_type in enumerate(required_op_types):
      ops_map['op_{:03d}'.format(i)] = op_type
    inv_ops_map = {v: k for k, v in ops_map.items()}
    template_vars = {}
    template_vars['model_name'] = ugraph.name
    template_vars['header_file'] = '{}.hpp'.format(ugraph.name)
    template_vars['meta_data_size'] = self._compute_meta_data_size(new_ugraph)
    template_vars['ram_data_size'] = self._compute_ram_data_size(new_ugraph)
    template_vars['placeholders'] = placeholders
    template_vars['placeholders_var_map'] = placeholders_var_map
    template_vars['ops_map'] = ops_map
    template_vars['inv_ops_map'] = inv_ops_map
    # 2. prepare ops and tensors
    # 3. generate computing logic
    # 4. write files
    # generate inline arrays
    # generate the computation function
    src_fname = self.src_fname == 'None' and '{}.cpp'.format(ugraph.name) or self.src_fname
    with open(src_fname, 'w') as fid:
      template = env.get_template('snippets/rearch/simple.cpp')
      fid.write(template.render(**template_vars))
    with open(template_vars['header_file'], 'w') as fid:
      template = env.get_template('snippets/rearch/simple.hpp')
      fid.write(template.render(**template_vars))

  @class_property
  def default_config(cls):
    config = {}
    config['src_fname'] = 'None'
    config['params_dir'] = 'data'
    config['embed_params_dir'] = '/fs/data'
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
    config['debug_cmt'] = False
    return config

  def _compute_meta_data_size(self, ugraph):
    if self.meta_data_pool_size == 'auto':
      # TODO: compute actual meta data size
      size = 256
    else:
      size = self.meta_data_pool_size
    return size

  def _compute_ram_data_size(self, ugraph):
    if self.ram_data_pool_size == 'auto':
      # TODO: compute actual ram data size
      size = 256
    else:
      size = self.ram_data_pool_size
    return size