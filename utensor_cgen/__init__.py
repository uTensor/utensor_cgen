"""
  isort:skip_file
"""
import sys

import pkg_resources

# https://github.com/google/or-tools/issues/1830
# we need to import ortools before tensorflow
from ortools.sat.python import cp_model as _

from utensor_cgen._extensions import _ExtensionsLoader

__version__ = (
  pkg_resources
  .get_distribution('utensor_cgen')
  .version
)
sys.modules['utensor_cgen.extensions'] = _ExtensionsLoader()
