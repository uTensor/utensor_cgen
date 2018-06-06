from .optimizer import RefCntOptimizer
from .ns_transformer import DropoutTransformer, BatchNormTransformer
from .quantize import QuantizeTransformer
from .base import Transformer
from utensor_cgen.utils import NamescopedKWArgsParser

class TransformerPipeline(object):

  _TRANSFORMER_MAP = {
    'refcnt': RefCntOptimizer,
    'dropout': DropoutTransformer,
    'batch_norm': BatchNormTransformer,
    'quantize': QuantizeTransformer
  }

  def __init__(self, methods, kwargs):
    """
    kwargs is a dict of following format:
    {
      "<name_scope>__kwargname": kwarg_value,
      ....
    }
    where <name_scope> is the KWARGS_NAMESCOPE of 
    desired transformer.
    
    ex:
    {
      'refcnt__kwarg': 3  # this is kwarg for RefCntOptimizer
    }
    """
    self._pipeline = []
    for method in methods:
      trans_cls = self._TRANSFORMER_MAP[method]
      trans_name = trans_cls.KWARGS_NAMESCOPE
      parser = NamescopedKWArgsParser(trans_name, kwargs)
      transformer = trans_cls(**parser.as_dict())
      self._pipeline.append(transformer)
  
  def transform(self, ugraph):
    for transformer in self._pipeline:
      ugraph = transformer.transform(ugraph)
    return ugraph

  @classmethod
  def all_transform_methods(cls):
    return list(cls._TRANSFORMER_MAP.keys())
  
  @classmethod
  def register_transformer(cls, method, trans_cls, overwrite=False):
    assert issubclass(trans_cls, Transformer), \
      "expecting Transformer type, get %s" % trans_cls
    assert method not in cls._TRANSFORMER_MAP or overwrite, \
      "Registering existing transformer without overwriting"
    cls._TRANSFORMER_MAP[name] = trans_cls