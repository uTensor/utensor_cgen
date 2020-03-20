import sys
import re
import importlib
from types import ModuleType

class _ExtensionsLoader(ModuleType):
  _ext_cache = {}
  _dunder_pattern = re.compile(r'__[A-Za-z0-9_]+__')

  def __init__(self):
    super(_ExtensionsLoader, self).__init__('utensor_cgen.extensions', {})
    
  def __getattr__(self, ext_name):
    if self._dunder_pattern.match(ext_name):
      _mod = sys.modules['utensor_cgen._extensions']
      return getattr(_mod, ext_name)
    if ext_name not in self._ext_cache:
      ext_mod = importlib.import_module(
          'utensor_{}'.format(ext_name)
      )
      self._ext_cache[ext_name] = ext_mod
      sys.modules['utensor_cgen.extensions.{}'.format(ext_name)] = ext_mod
    return self._ext_cache[ext_name]
    
  def __dir__(self):
    return self._ext_cache.keys()
