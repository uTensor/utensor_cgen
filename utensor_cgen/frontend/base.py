from abc import ABCMeta, abstractmethod

class Parser(object):
  __metaclass__ = ABCMeta

  @classmethod
  @abstractmethod
  def parse(cls, fname, outupt_nodes):
      raise RuntimeError('abstract parse method involded')
