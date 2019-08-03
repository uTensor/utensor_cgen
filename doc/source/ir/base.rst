:mod:`ir.base`
^^^^^^^^^^^^^^

Concepts
--------

Conceptually, :class:`.uTensorGraph` works like a container of
:class:`.TensorInfo` and :class:`.OperationInfo`.

That is, as long as the :class:`.TensorInfo` and :class:`.OperationInfo`
is not dangling, they always belong to some :class:`.uTensorGraph`
which should be accessible via **ugraph** property of both
:class:`.TensorInfo` and :class:`.OperationInfo`.

Otherwise, the graph, tensor or op is dangling.

For short,
we'll refer an instance of :class:`.OperationInfo` as an *op* or a *node*.

Developer Note
--------------

Keep following tips in mind if you try to directly manipulate
the state of a graph, op or tensor

- **No shallow copy allowed**
- About :class:`.TensorInfo`:

  - **name**, **op_name**, **dtype** and **shape** are the only
    attributes you can manipulate directly

    - when you do, you are responsible to make sure these values
      are valid
  - make sure the **op_name** is set as the :class:`.OperationInfo`'s
    name which gererates this tensor

    - **op_name** is the identifier used to retrieve the **op** in
      the graph
    - incorrect **op_name** may make the tensor dangling
  - When you try to transfer a tensor from one graph to the other,
    use `move_into <#utensor_cgen.ir.base.TensorInfo.move_into>`_
    method instead.
- About :class:`.OperationInfo`:

  - hello


Module members
--------------

.. autoapimodule:: utensor_cgen.ir.base
    :members:
