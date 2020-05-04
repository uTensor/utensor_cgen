import pickle

from utensor_cgen.logger import logger
from utensor_cgen.transformer.pipeline import TransformerPipeline
from utensor_cgen.utils import class_property

from .base import BackendPart

__all__ = ['PipelineTransformer']

class PipelineTransformer(BackendPart):
  TARGET = 'generic'
  PART = 'pipeline_transformer'

  def __init__(self, config):
    self.transformer = TransformerPipeline(
      methods=self.config['transform_methods']
    )
    self.trans_methods = self.config['transform_methods']
    self.save_graph = self.config['save_graph']

  def transform(self, ugraph):
    logger.info("Transforming graph: %s", ugraph.name)
    logger.info("Transform pipeline: %s", ' -> '.join(self.trans_methods))
    self._check_non_quantized(ugraph)
    new_ugraph = self.transformer.transform(ugraph)
    new_ugraph.name = ugraph.name
    logger.info('Graph transormation done')
    if self.save_graph:
      logger.info('Saving transformed graph')
      pkl_fname = "{}_transformed.pkl".format(ugraph.name)
      with open(pkl_fname, 'wb') as fid:
        pickle.dump(new_ugraph, fid)
      logger.info('{} saved'.format(pkl_fname))
    return new_ugraph
  
  @classmethod
  def _check_non_quantized(cls, ugraph):
    is_quantized = False
    quant_ops = set([
      "Dequantize", "QuantizedMaxPool",
      "QuantizeV2", "QuantizedMatMul",
      "QuantizedRelu", "QuantizedAdd",
      "RequantizationRange",
      "Requantize",
      "QuantizedReshape",
      "QuantizedConv2D"
    ])
    for op_info in ugraph.ops_info.values():
      if op_info.op_type in quant_ops:
        is_quantized = True
        break
    if is_quantized:
      logger.warning((
        "Expecting non-quantized graph, "
        "graph transformation/optimization might not work properly"
      ))

  @class_property
  def default_config(cls):
    return {
      'save_graph': False,
      'transform_methods': [
        "dropout(name_pattern=r'(dropout[_\w\d]*)/.*')",
        "linear_reorder",
        # these methods are deprecated
        # "quantize",
        # "conv_pool",
        "inline",
        "biasAdd",
        "remove_id_op",
        "fake_gather_v2",
        "refcnt",
      ]
    }
