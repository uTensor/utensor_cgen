from .base import Transformer


class TransformerPipeline(object):

  TRANSFORMER_MAP = {}

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
