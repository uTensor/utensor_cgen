from .base import Transformer
from .conv_pool import ConvPoolTransformer
from .graph_viz import GraphVizTransformer
from .linear_reoder import LinearReorderTransformerV2
from .ns_transformer import (BatchNormTransformer, BiasAddTransformer,
                             DropoutTransformer, DropoutTransformerV2,
                             FakeGatherV2Transformer, InlineTransformer,
                             TensorLifeProbe)
from .optimizer import IdOpRemoveOptimizer, RefCntOptimizer
from .quantize import QuantizeTransformer


class TransformerPipeline(object):

  TRANSFORMER_MAP = {
    RefCntOptimizer.METHOD_NAME: RefCntOptimizer,
    DropoutTransformer.METHOD_NAME: DropoutTransformer,
    DropoutTransformerV2.METHOD_NAME: DropoutTransformerV2,
    BatchNormTransformer.METHOD_NAME: BatchNormTransformer,
    QuantizeTransformer.METHOD_NAME: QuantizeTransformer,
    InlineTransformer.METHOD_NAME: InlineTransformer,
    BiasAddTransformer.METHOD_NAME: BiasAddTransformer,
    IdOpRemoveOptimizer.METHOD_NAME: IdOpRemoveOptimizer,
    GraphVizTransformer.METHOD_NAME: GraphVizTransformer,
    FakeGatherV2Transformer.METHOD_NAME: FakeGatherV2Transformer,
    ConvPoolTransformer.METHOD_NAME: ConvPoolTransformer,
    LinearReorderTransformerV2.METHOD_NAME: LinearReorderTransformerV2,
    TensorLifeProbe.METHOD_NAME: TensorLifeProbe,
  }

  def __init__(self, methods):
    """
    methods : list
      list of tuples, (transform_name, kwargs)
    """
    self._pipeline = []
    for method, kwargs in methods:
      trans_cls = self.TRANSFORMER_MAP.get(method, None)
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
    return list(cls.TRANSFORMER_MAP.keys())
  
  @classmethod
  def register_transformer(cls, trans_cls, overwrite=False):
    assert issubclass(trans_cls, Transformer), \
      "expecting Transformer type, get %s" % trans_cls
    assert trans_cls.METHOD_NAME not in cls.TRANSFORMER_MAP or overwrite, \
      "Registering existing transformer without overwriting"
    cls.TRANSFORMER_MAP[trans_cls.METHOD_NAME] = trans_cls
