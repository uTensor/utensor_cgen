from abc import abstractmethod

from utensor_cgen.utils import (MUST_OVERWRITE, Configuration, class_property,
                                is_abstract, parse_toml)


class _BackendBase(object):

  COMPONENT = 'backend'

  def __new__(cls, config, *args, **kwargs):
    self = object.__new__(cls)
    for name in dir(self):
      if name.startswith('_validate'):
        validator = getattr(self, name)
        validator(config, *args, **kwargs)
    if isinstance(config, dict):
      config = Configuration(cls.default_config, config)
    elif config is None:
      config = Configuration(cls.default_config, {})
    self._config = config
    return self

  @abstractmethod
  def apply(self, ugraph):
    """Applying side-effect to ugraph

    Any backend part that implement apply method can create side-effect on given graph,
    such as adding attribute or creating files.
    """
    raise NotImplementedError('base apply method invoked: %s' % self)

  @abstractmethod
  def transform(self, ugraph):
    """Transform Graph

    transform should not create side-effect on the given graph and should
    return a new ugraph that is the result of transformation applied to the
    given ugraph.
    """
    raise NotImplementedError('base transform method invoked: %s' % self)

  @class_property
  def default_config(cls):
    raise NotImplementedError('All backends should overwrite default config')

  def __call__(self, *args, **kwargs):
    return self.apply(*args, **kwargs)

  @property
  def config(self):
    return self._config

  def _validate_config(self, config, *args, **kwargs):
    assert isinstance(config, (dict, Configuration, type(None))), \
      'expecting {}, get {}'.format(dict, type(config))

  def _validate_abstracts(self, config, *args, **kwargs):
    if is_abstract(self.apply) and is_abstract(self.transform):
      raise ValueError('must overwrite at least one of apply or transorm: %s' % self)


class Backend(_BackendBase):
  """
  - Constrcutor signature must be ``__init__(self, config, *args, **kwargs)``
    - ``config`` should be a dictionay
    - It will run through various check in ``__new__``, so it's better to access the value
    of config via ``self.config``, which is an instance of ``Configuration``
    - It will make sure if users do not provide the value required, default one will be used
  - You must at least implement one of ``apply`` or ``transform`` method
    - ``apply`` will introduce side-effect on given ugraph **in place** and return nothing
    - ``transform`` will create a new ugraph, applying side-effect on new ugraph and return it
  """

  TARGET = MUST_OVERWRITE

  @classmethod
  def _validate_target(cls, config, *args, **kwargs):
    if cls.TARGET is MUST_OVERWRITE:
      raise ValueError(
        'Every Backend must overwrite TARGET attribute: {}'.format(cls)
      )

  @classmethod
  def from_file(cls, file_or_path, *args, **kwargs):
    config = parse_toml(file_or_path)[cls.TARGET][cls.COMPONENT]
    return cls(config, *args, **kwargs)

  @classmethod
  def from_config(cls, config, *args, **kwargs):
    return cls(config, *args, **kwargs)


class BackendPart(Backend):

  PART = MUST_OVERWRITE

  @classmethod
  def _validate_part(cls, config, *args, **kwargs):
    if cls.PART is MUST_OVERWRITE:
      raise ValueError(
        'Every BackendPart must overwrite PART attribute: {}'.format(cls)
      )

  @classmethod
  def from_config(cls, config, *args, **kwargs):
    default_config = cls.default_config
    if cls.TARGET in config:
      config = config[cls.TARGET]
    if cls.COMPONENT in config:
      config = config[cls.COMPONENT]
    if cls.PART in config:
      config = config[cls.PART]
    default_config.update(config)
    return cls(default_config, *args, **kwargs)

  @classmethod
  def from_file(cls, file_or_path, *args, **kwargs):
    config = parse_toml(file_or_path)[cls.TARGET][cls.COMPONENT][cls.PART]
    return cls(config, *args, **kwargs)
