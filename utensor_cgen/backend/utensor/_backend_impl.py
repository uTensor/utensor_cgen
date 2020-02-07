from utensor_cgen.backend.base import Backend
from utensor_cgen.utils import (
  class_property, parse_toml,
  LazyLoader, LazyAttrib,
  Configuration,
)

code_generator = LazyLoader(submod_name='backend.utensor.code_generator')
_graph_lower = LazyLoader(submod_name='backend.utensor._graph_lower')
uTensorLegacyCodeGenerator = LazyAttrib(code_generator, 'uTensorLegacyCodeGenerator')
uTensorRearchCodeGenerator = LazyAttrib(code_generator, 'uTensorRearchCodeGenerator')
uTensorGraphLower = LazyAttrib(_graph_lower, 'uTensorGraphLower')

del code_generator, _graph_lower

class uTensorBackend(Backend):

  TARGET = 'utensor'

  def __init__(self, config, code_generator=None, graph_lower=None):
    default_config = self.default_config[self.TARGET][self.COMPONENT]
    config = Configuration(default_config, config.get(
      self.TARGET,
      {self.COMPONENT: {}}
    ).get(
      self.COMPONENT, {}
      )
    )
    if code_generator is None:
      if config['legacy-api']:
        part_config = config[uTensorLegacyCodeGenerator.PART]
        code_generator = uTensorLegacyCodeGenerator(config=part_config)
      else:
        part_config = config[uTensorRearchCodeGenerator.PART]
        code_generator = uTensorRearchCodeGenerator(config=part_config)
    if graph_lower is None:
      part_config = config[uTensorGraphLower.PART]
      graph_lower = uTensorGraphLower(config=part_config)
    self._graph_lower = graph_lower
    self._code_generator = code_generator
  
  @class_property
  def default_config(cls):
    config = {}
    config[cls.TARGET] = {}
    config[cls.TARGET][cls.COMPONENT] = {}
    config[cls.TARGET][cls.COMPONENT]['legacy-api'] = True
    config[cls.TARGET][cls.COMPONENT][uTensorLegacyCodeGenerator.PART] = uTensorLegacyCodeGenerator.default_config
    config[cls.TARGET][cls.COMPONENT][uTensorRearchCodeGenerator.PART] = uTensorRearchCodeGenerator.default_config
    config[cls.TARGET][cls.COMPONENT][uTensorGraphLower.PART] = uTensorGraphLower.default_config
    return config

  def apply(self, ugraph):
    lower_ugraph = self._graph_lower.apply(ugraph)
    self._code_generator.apply(lower_ugraph)
    return lower_ugraph

  def __call__(self, ugraph):
    return self.apply(ugraph)

  @classmethod
  def from_file(cls, path_or_file):
    config = parse_toml(path_or_file)
    return cls(config=config)
