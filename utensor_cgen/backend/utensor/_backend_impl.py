from utensor_cgen.backend.base import Backend
from utensor_cgen.utils import (LazyAttrib, LazyLoader, class_property,
                                parse_toml)

code_generator = LazyLoader(submod_name='backend.utensor.code_generator')
transformer = LazyLoader(submod_name='backend.transformer')
_op_lower = LazyLoader(submod_name='backend.utensor._graph_lower._op_lower')
generic_graph_lower = LazyLoader(submod_name='backend.graph_lower')
uTensorLegacyCodeGenerator = LazyAttrib(code_generator, 'uTensorLegacyCodeGenerator')
uTensorRearchCodeGenerator = LazyAttrib(code_generator, 'uTensorRearchCodeGenerator')
uTensorLegacyGraphLower = LazyAttrib(_op_lower, 'uTensorLegacyGraphLower')
uTensorRearchGraphLower = LazyAttrib(_op_lower, 'uTensorRearchGraphLower')
TensorAllocationPlanner = LazyAttrib(generic_graph_lower, 'TensorAllocationPlanner')
PipelineTransformer = LazyAttrib(transformer, 'PipelineTransformer')

del code_generator, _op_lower, generic_graph_lower

class uTensorBackend(Backend):

  TARGET = 'utensor'

  def __init__(
    self,
    config,
    code_generator=None,
    graph_transformer=None,
    graph_op_lower=None,
    graph_alloc_lower=None,
  ):
    config = self.config[self.TARGET][self.COMPONENT]
    if config['legacy-api']:
      code_generator = code_generator or uTensorLegacyCodeGenerator(config=config[uTensorLegacyCodeGenerator.PART].to_dict())
      graph_op_lower = graph_op_lower or uTensorLegacyGraphLower(config=config[uTensorLegacyGraphLower.PART].to_dict())
      graph_alloc_lower = graph_alloc_lower or TensorAllocationPlanner(config=config[TensorAllocationPlanner.PART].to_dict())
    else:
      code_generator = code_generator or uTensorRearchCodeGenerator(config=config[uTensorRearchCodeGenerator.PART].to_dict())
      graph_op_lower = graph_op_lower or uTensorRearchGraphLower(config=config[uTensorRearchGraphLower.PART].to_dict())
      graph_alloc_lower = TensorAllocationPlanner(config=config[TensorAllocationPlanner.PART].to_dict())
    graph_transformer = graph_transformer or PipelineTransformer(config=config[PipelineTransformer.PART].to_dict())
    self._graph_op_lower = graph_op_lower
    self._graph_transformer = graph_transformer
    self._graph_alloc_lower = graph_alloc_lower
    self._code_generator = code_generator
  
  @class_property
  def default_config(cls):
    config = {}
    config[cls.TARGET] = {}
    config[cls.TARGET][cls.COMPONENT] = {}
    config[cls.TARGET][cls.COMPONENT]['legacy-api'] = False
    config[cls.TARGET][cls.COMPONENT][TensorAllocationPlanner.PART] = TensorAllocationPlanner.default_config
    config[cls.TARGET][cls.COMPONENT][uTensorLegacyCodeGenerator.PART] = uTensorLegacyCodeGenerator.default_config
    config[cls.TARGET][cls.COMPONENT][uTensorLegacyGraphLower.PART] = uTensorLegacyGraphLower.default_config
    config[cls.TARGET][cls.COMPONENT][uTensorRearchCodeGenerator.PART] = uTensorRearchCodeGenerator.default_config
    config[cls.TARGET][cls.COMPONENT][uTensorRearchGraphLower.PART] = uTensorRearchGraphLower.default_config
    config[cls.TARGET][cls.COMPONENT][PipelineTransformer.PART] = PipelineTransformer.default_config
    return config

  def apply(self, ugraph):
    # 1. graph optimization
    opt_ugraph = self._graph_transformer.transform(ugraph)
    # 2. lowering to target specific graph
    self._graph_op_lower.apply(opt_ugraph)
    # 3. apply memory allocation planner
    self._graph_alloc_lower.apply(opt_ugraph)
    # 4. generate target files
    self._code_generator.apply(opt_ugraph)

  def __call__(self, ugraph):
    return self.apply(ugraph)

  @classmethod
  def from_file(cls, path_or_file):
    config = parse_toml(path_or_file)
    return cls(config=config)
