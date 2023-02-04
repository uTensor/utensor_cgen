"""
  isort:skip_file
"""
# https://github.com/google/or-tools/issues/1830
# we need to import ortools before tensorflow
from ortools.sat.python import cp_model as _

from ._version import __version__
