from abc import ABCMeta, abstractmethod
from copy import deepcopy

import attr

from utensor_cgen.utils import MUST_OVERWRITEN


class Morphism(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def apply(self, from_op):
    raise RuntimeError('abstract transform invoked')


@attr.s
class TypedMorphism(Morphism):
  from_op_type = MUST_OVERWRITEN

  def __attrs_post_init__(self):
    if self.from_op_type is MUST_OVERWRITEN:
      raise ValueError(
        "must overwrite {}.from_op_type".format(type(self).__name__)
      )

class GenericMorphism(Morphism):
  pass

class IdentityMorphism(GenericMorphism):

  def apply(self, from_op):
    return from_op

@attr.s
class Const2InlineMorphism(TypedMorphism):
  from_op_type = 'Const'

  def apply(self, from_op):
    new_op = deepcopy(from_op, memo={'ugraph': from_op.ugraph})
    new_op.op_type = 'Inline'
    return new_op


@attr.s
class Inline2ConstMorphism(TypedMorphism):
  from_op_type = 'Inline'

  def apply(self, from_op):
    new_op = deepcopy(from_op, memo={'ugraph': from_op.ugraph})
    new_op.op_type = 'Const'
    return new_op
