from utensor_cgen.utils import MUST_OVERWRITEN

class FrontendSelector(object):
  _parser_map = {}

  @classmethod
  def register(cls, target_exts):
    def _register(parser_cls):
      for ext in target_exts:
        if ext in cls._parser_map:
          raise ValueError("duplicate file ext detected: %s" % ext)
      cls._parser_map[ext] = parser_cls
      return parser_cls

    return _register

  @classmethod
  def select_parser(cls, file_ext):
    parser_cls = cls._parser_map.get(file_ext, None)
    if parser_cls is None:
      raise RuntimeError("unknown model file ext found: %s" % file_ext)
    return parser_cls

from . import onnx as _
from . import tensorflow as _