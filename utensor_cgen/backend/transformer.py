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
    if not self._check_generic(ugraph):
      raise ValueError(
        'the given graph is not generic:\n{}'.format(ugraph)
      )
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
  def _check_generic(cls, ugraph):
    # TODO: do the real check once we have full list of generic ops
    return True

  @class_property
  def default_config(cls):
    return {
      'save_graph': False,
      'transform_methods': [
        "dropout(name_pattern=r'(dropout[_\w\d]*)/.*')",
        # "linear_reorder", # FIXME: uncomment this after fixing Matcher
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
