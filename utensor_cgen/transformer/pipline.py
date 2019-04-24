from utensor_cgen.utils import NamescopedKWArgsParser

from .base import Transformer
from .cmsis_nn import CMSIS_NN_Transformer
from .ns_transformer import (BatchNormTransformer, DropoutTransformer,
                             InlineTransformer, BiasAddTransformer)
from .optimizer import IdOpRemoveOptimizer, RefCntOptimizer
from .quantize import QuantizeTransformer


class TransformerPipeline(object):

  _TRANSFORMER_MAP = {
    RefCntOptimizer.METHOD_NAME: RefCntOptimizer,
    DropoutTransformer.METHOD_NAME: DropoutTransformer,
    BatchNormTransformer.METHOD_NAME: BatchNormTransformer,
    QuantizeTransformer.METHOD_NAME: QuantizeTransformer,
    InlineTransformer.METHOD_NAME: InlineTransformer,
    BiasAddTransformer.METHOD_NAME: BiasAddTransformer,
    CMSIS_NN_Transformer.METHOD_NAME: CMSIS_NN_Transformer,
    IdOpRemoveOptimizer.METHOD_NAME: IdOpRemoveOptimizer,
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
  
  @property
  def pipeline(self):
    return self._pipeline

  @classmethod
  def all_transform_methods(cls):
    return list(cls._TRANSFORMER_MAP.keys())
  
  @classmethod
  def register_transformer(cls, trans_cls, overwrite=False):
    assert issubclass(trans_cls, Transformer), \
      "expecting Transformer type, get %s" % trans_cls
    assert trans_cls.METHOD_NAME not in cls._TRANSFORMER_MAP or overwrite, \
      "Registering existing transformer without overwriting"
    cls._TRANSFORMER_MAP[trans_cls.METHOD_NAME] = trans_cls
