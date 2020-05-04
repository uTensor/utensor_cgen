import sys

import pkg_resources

# https://github.com/google/or-tools/issues/1830
# we need to import ortools before tensorflow
from ortools.sat.python import cp_model as _ # isort:skip
import tensorflow.compat.v1 as _tf # isort:skip

_tf.disable_v2_behavior() # isort:skip
_tf.disable_v2_tensorshape() # isort:skip

from utensor_cgen._extensions import _ExtensionsLoader # isort:skip

__version__ = (
  pkg_resources
  .get_distribution('utensor_cgen')
  .version
)
sys.modules['utensor_cgen.extensions'] = _ExtensionsLoader()

del _tf
