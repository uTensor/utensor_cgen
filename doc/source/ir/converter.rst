.. _converter:

:mod:`converter`
^^^^^^^^^^^^^^^^

Concepts
--------

This module defines the interface of a converter.

A converter is responsible for converting tensorflow/pytorch
types to or from generic types defined in `utensor_cgen </>`_

:class:`GenericConverter`
=========================

Any subclass of :class:`.GenericConverter` should overwrite
following methods/attributes:

1. :meth:`.GenericConverter.get_generic_value`

    - should reture value of type as **__utensor_generic_type__**
2. :class:`.GenericConverter`\ **.__utensor_generic_type__**

    - defines the return type of :meth:`.GenericConverter.get_generic_value`

:class:`TFConverterMixin`
=========================

Any subclass of :class:`.TFConverterMixin` should overwrite
following methods/attributes:

1. :meth:`.TFConverterMixin.get_tf_value`

    - should return a value of type as **__tfproto_type__**
2. :class:`.TFConverterMixin`\ **.__tfproto_type__**

    - defines the return type of :meth:`.TFConverterMixin.get_tf_value`

A Qualified Converter in `utensor_cgen </>`_
============================================

A qualified converter type should be subclass of both
:class:`.GenericConverter` and :class:`.TFConverterMixin`

That is, a qualified converter should be able to convert
a :mod:`tensorflow` protobuf type to a `utensor_cgen </>`_
generic data type.

Such convertion is defined by its :attr:`__utensor_generic_type__`
and :attr:`__tfproto_type__` attributes.

Given a value of type :attr:`__tfproto_type__`, :meth:`!get_generic_value`
will convert it to an equivalent value of type :attr:`__utensor_generic_type__`.
:meth:`!get_tf_value` is hence an inverse function of :meth:`!get_generic_value`


Module Members
--------------

.. automodule:: utensor_cgen.ir.converter
    :members: ConverterDispatcher, GenericConverter, TFConverterMixin
