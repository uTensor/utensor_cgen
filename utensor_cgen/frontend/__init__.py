import importlib
import os

from .base import Parser


class FrontendSelector(object):
  _parser_map = {}
  _setuped = False

  @classmethod
  def register(cls, target_exts):
    def _register(parser_cls):
      if not issubclass(parser_cls, Parser):
        raise TypeError('incorrect parser type for registration')
      for ext in target_exts:
        if ext in cls._parser_map:
          raise ValueError("duplicate file ext detected: %s" % ext)
        cls._parser_map[ext] = parser_cls
      return parser_cls

    return _register

  @classmethod
  def select_parser(cls, file_ext):
    cls._setup()
    parser_cls = cls._parser_map.get(file_ext, None)
    if parser_cls is None:
      raise RuntimeError("unknown model file ext found: %s" % file_ext)
    return parser_cls
  
  @classmethod
  def _setup(cls):
    """
    Find all .py files under `utensor_cgen.frontend` and import it
    to register all parsers
    """
    if cls._setuped:
      return
    root_dir = os.path.dirname(__file__)
    _, _, files = next(os.walk(root_dir))
    for file in files:
      fname, ext = os.path.splitext(file)
      if fname not in ['__init__', 'base'] and ext == ".py":
        mod_name = 'utensor_cgen.frontend.%s' % fname
        importlib.import_module(mod_name)
    cls._setuped = True
