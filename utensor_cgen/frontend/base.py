from abc import ABCMeta, abstractmethod
from copy import deepcopy

from utensor_cgen.utils import parse_tensor_name


class Parser(object):
  __metaclass__ = ABCMeta

  @classmethod
  @abstractmethod
  def parse(cls, fname, outupt_nodes):
      raise RuntimeError('abstract parse method involded')
