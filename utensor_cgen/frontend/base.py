from abc import ABCMeta, abstractmethod

from utensor_cgen.utils import Configuration, class_property


class Parser(object):
  __metaclass__ = ABCMeta

  def __new__(cls, config):
    if isinstance(config, dict):
      config = Configuration(defaults=cls.default_config, user_config=config)
    if not isinstance(config, Configuration):
      raise ValueError(
        f'invalid config. Should be dict or Configuration, get {type(config)}'
      )
    self = object.__new__(cls)
    self._config = config
    return self

  @property
  def config(self):
    return self._config

  @class_property
  def default_config(cls):
    return {}

  @classmethod
  @abstractmethod
  def parse(cls, fname, output_nodes=None, model_name=None, **kwargs):
      raise RuntimeError('abstract parse method involded')
