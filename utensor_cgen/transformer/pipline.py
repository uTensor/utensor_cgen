from utensor_cgen.utils import NamescopedKWArgsParser

from .base import Transformer
from .cmsis_nn import CMSIS_NN_Transformer
from .ns_transformer import (BatchNormTransformer, DropoutTransformer,
                             InlineTransformer, BiasAddTransformer, TensorLifeProbe)
from .optimizer import IdOpRemoveOptimizer, RefCntOptimizer
from .quantize import QuantizeTransformer
from .graph_viz import GraphVizTransformer

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
    GraphVizTransformer.METHOD_NAME: GraphVizTransformer,
    TensorLifeProbe.METHOD_NAME: TensorLifeProbe,
  }

  def __init__(self, methods):
    """
    methods : list
      list of tuples, (transform_name, kwargs)
    """
    self._pipeline = []
    for method, kwargs in methods:
      trans_cls = self._TRANSFORMER_MAP.get(method, None)
      if trans_cls is None:
        raise ValueError("Unknown transformation method: {}".format(method))
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
    return list(cls._TRANSFORMER_MAP.keys())
  
  @classmethod
  def register_transformer(cls, trans_cls, overwrite=False):
    assert issubclass(trans_cls, Transformer), \
      "expecting Transformer type, get %s" % trans_cls
    assert trans_cls.METHOD_NAME not in cls._TRANSFORMER_MAP or overwrite, \
      "Registering existing transformer without overwriting"
    cls._TRANSFORMER_MAP[trans_cls.METHOD_NAME] = trans_cls
