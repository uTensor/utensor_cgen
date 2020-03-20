import re
from ast import literal_eval

from .base import Transformer


class TransformerPipeline(object):

  TRANSFORMER_MAP = {}

  _trans_name_patrn = re.compile(r"(\w[\w]*)\(?")

  def __init__(self, methods):
    """
    methods : list
      list of tuples of type Tuple[Type[Transformer], dict] or a string expression
      of the transformer such as 'dropout(name_pattern=r"(dropout[_\w\d]*)/.*")'
    """
    self._pipeline = []
    for method_or_str in methods:
      if isinstance(method_or_str, str):
        method, kwargs = self._parse_expr(method_or_str)
        trans_cls = self.TRANSFORMER_MAP.get(method, None)
        if trans_cls is None:
          raise ValueError("Unknown transformation method: {}".format(method))
      else:
        trans_cls, kwargs = method_or_str
      if not issubclass(trans_cls, Transformer):
        raise TypeError("expecting subclass of {}, get {}".format(Transformer, trans_cls))
      transformer = trans_cls(**kwargs)
      self._pipeline.append(transformer)
  
  def transform(self, ugraph):
    for transformer in self._pipeline:
      ugraph = transformer.transform(ugraph)
    return ugraph
  
  @property
  def pipeline(self):
    return self._pipeline

  @classmethod
  def all_transform_methods(cls):
    return list(cls.TRANSFORMER_MAP.keys())
  
  @classmethod
  def register_transformer(cls, trans_cls=None, overwrite=False):
    def register(trans_cls):
      if not issubclass(trans_cls, Transformer):
        raise ValueError("expecting Transformer type, get %s" % trans_cls)
      if not overwrite and trans_cls.METHOD_NAME in cls.TRANSFORMER_MAP:
        raise RuntimeError("Registering existing transformer without overwriting")
      cls.TRANSFORMER_MAP[trans_cls.METHOD_NAME] = trans_cls
      return trans_cls
  
    if trans_cls is None:
      return register
    return register(trans_cls)
  
  @classmethod
  def _parse_expr(cls, expr):
    trans_match = cls._trans_name_patrn.match(expr)
    if not trans_match:
      raise ValueError("Invalid args detected: {}".format(expr))
    trans_name = trans_match.group(1)
    _, end = trans_match.span()
    if end == len(expr):
      kwargs = {}
    else:
      if not expr.endswith(")"):
        raise ValueError("parentheses mismatch: {}".format(expr))
      kwargs = cls._get_kwargs(expr[end:-1])
    return trans_name, kwargs

  @classmethod
  def _get_kwargs(cls, kws_str):
    kw_arg_strs = [s.strip() for s in kws_str.split(',')]
    kwargs = {}
    for kw_str in kw_arg_strs:
      name, v_str = kw_str.split('=')
      value = literal_eval(v_str)
      kwargs[name] = value
    return kwargs
