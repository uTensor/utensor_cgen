from utensor_cgen.utils import MUST_OVERWRITEN


class LegalizerBase(object):

  TARGET = MUST_OVERWRITEN
  COMPONET = 'legalizer'

  def __new__(cls, config):
    if not isinstance(config, dict):
      raise TypeError(
        'expecting dict as config, get {}'.format(type(config))
      )
    if cls.TARGET is MUST_OVERWRITEN:
      raise ValueError('cls.TARGET must be overwriten')
    self = object.__new__(cls)
    self._config = config
    return self

  def legalize(self, ugraph):
    ugraph = self.legalize_ops(ugraph)
    ugraph = self.legalize_dtype(ugraph)
    return ugraph

  def legalize_ops(self, ugraph):
    raise NotImplementedError('abstract ops legalizer get called')

  def legalize_dtype(self, ugraph):
    raise NotImplementedError('abstract dtype legalizer get called')
