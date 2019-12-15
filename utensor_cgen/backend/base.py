from utensor_cgen.utils import MUST_OVERWRITEN, class_property, parse_toml


class _BackendBase(object):

  COMPONENT = 'backend'

  def __new__(cls, config, *args, **kwargs):
    self = object.__new__(cls)
    for name in dir(self):
      if name.startswith('_validate'):
        validator = getattr(self, name)
        validator(config, *args, **kwargs)
    self._config = config
    return self

  def apply(self, ugraph):
    raise NotImplementedError('all backend object must overwrite apply method')

  @class_property
  def default_config(cls):
    raise RuntimeError('No default configuration for {}'.format(cls))

  def __call__(self, *args, **kwargs):
    return self.apply(*args, **kwargs)

  @property
  def config(self):
    return self._config

  def _validate_config(self, config):
    assert isinstance(config, dict), \
      'expecting {}, get {}'.format(dict, type(config))


class Backend(_BackendBase):

  TARGET = MUST_OVERWRITEN

  @classmethod
  def _validate_target(cls, config, *args, **kwargs):
    if cls.TARGET is MUST_OVERWRITEN:
      raise ValueError(
        'Every Backend must overwrite TARGET attribute: {}'.format(cls)
      )

  @classmethod
  def from_file(cls, file_or_path, *args, **kwargs):
    config = parse_toml(file_or_path)[cls.TARGET][cls.COMPONENT]
    return cls(config, *args, **kwargs)

  @classmethod
  def from_config(cls, config, *args, **kwargs):
    default_config = cls.default_config
    if cls.TARGET in config:
      config = config[cls.TARGET]
    default_config.update(config)
    return cls(default_config, *args, **kwargs)

class BackendPart(Backend):

  PART = MUST_OVERWRITEN

  @classmethod
  def _validate_part(cls, config, *args, **kwargs):
    if cls.PART is MUST_OVERWRITEN:
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
