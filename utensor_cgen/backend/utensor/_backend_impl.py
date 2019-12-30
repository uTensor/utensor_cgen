from utensor_cgen.backend.base import Backend
from utensor_cgen.utils import class_property, parse_toml, LazyLoader, LazyAttrib

_code_generator = LazyLoader(submod_name='backend.utensor._code_generator')
_graph_lower = LazyLoader(submod_name='backend.utensor._graph_lower')
uTensorCodeGenerator = LazyAttrib(_code_generator, 'uTensorCodeGenerator')
uTensorGraphLower = LazyAttrib(_graph_lower, 'uTensorGraphLower')

del _code_generator, _graph_lower

class uTensorBackend(Backend):

  TARGET = 'utensor'

  def __init__(self, config, code_generator=None, graph_lower=None):
    final_config = self.default_config[self.TARGET]
    if config:
      final_config.update(config[self.TARGET])
    if code_generator is None:
      part_config = final_config[self.COMPONENT][uTensorCodeGenerator.PART]
      code_generator = uTensorCodeGenerator(config=part_config)
    if graph_lower is None:
      part_config = final_config[self.COMPONENT][uTensorGraphLower.PART]
      graph_lower = uTensorGraphLower(config=part_config)
    self._graph_lower = graph_lower
    self._code_generator = code_generator
  
  @class_property
  def default_config(cls):
    config = {}
    config[cls.TARGET] = {}
    config[cls.TARGET][cls.COMPONENT] = {}
    config[cls.TARGET][cls.COMPONENT][uTensorCodeGenerator.PART] = uTensorCodeGenerator.default_config
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
    code_generator = uTensorCodeGenerator.from_config(config)
    graph_lower = uTensorGraphLower.from_config(config)
    return cls(code_generator=code_generator, graph_lower=graph_lower)
