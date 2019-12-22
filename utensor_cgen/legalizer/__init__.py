from .base import LegalizerBase
from .tensorflow import GraphDefLegalizer


class Legalizer(object):
  LEGALIZER_MAP = {}

  @classmethod
  def register(cls, legalizer=None):
    def _register(legalizer_):
      if not issubclass(legalizer_, LegalizerBase):
        raise TypeError(
          'expecting subclass of {}, get {}'.format(LegalizerBase, legalizer_)
        )
      cls.LEGALIZER_MAP[legalizer_.TARGET] = legalizer_
      return legalizer_
    if legalizer is None:
      return _register
    else:
      return _register(legalizer)
  
  @classmethod
  def legalize(cls, ugraph, config=None):
    if config is None:
      config = {}
    legalizer_cls = cls.LEGALIZER_MAP.get(ugraph.lib_name)
    if legalizer_cls is None:
      raise ValueError(
        'graph of unsupported lib given: {}'.format(ugraph.lib_name)
      )
    legalizer = legalizer_cls(config)
    return legalizer.legalize(ugraph)

Legalizer.register(GraphDefLegalizer)
