import sys

import pkg_resources

from utensor_cgen._extensions import _ExtensionsLoader

__version__ = (
  pkg_resources
  .get_distribution('utensor_cgen')
  .version
)
sys.modules['utensor_cgen.extensions'] = _ExtensionsLoader()
