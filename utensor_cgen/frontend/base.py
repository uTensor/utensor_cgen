from abc import ABCMeta, abstractmethod


class Parser(object):
  __metaclass__ = ABCMeta

  def __new__(cls, config):
    if not isinstance(config, dict):
      raise ValueError('expecting dict as config, get {}'.format(type(config)))
    self = object.__new__(cls)
    self._config = config
    return self

  @property
  def config(self):
    return self._config

  @classmethod
  @abstractmethod
  def parse(cls, fname, outupt_nodes):
      raise RuntimeError('abstract parse method involded')
