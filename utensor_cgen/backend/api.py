from utensor_cgen.utils import class_property

from .base import Backend
from .utensor import uTensorBackend


class BackendManager(object):
  BACKENDS = {}

  @classmethod
  def get_backend(cls, name):
    if name not in cls.BACKENDS:
      raise ValueError('unknown backend name: %s' % name)
    return cls.BACKENDS[name]

  @classmethod
  def register(cls, backend_cls):
    if not issubclass(backend_cls, Backend):
      raise TypeError(
        'can only register subclass of %s: get %s' % (Backend, backend_cls)
      )
    cls.BACKENDS[backend_cls.TARGET] = backend_cls

  @class_property
  def backends(cls):
    return list(cls.BACKENDS.keys())

BackendManager.register(uTensorBackend)
