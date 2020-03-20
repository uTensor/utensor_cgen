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
    use :py:meth:`.TensorInfo.move_into` method.
  - passing a :class:`.uTensorGraph` as an argument to the constructor
    means this tensor is owned by the given graph.
- About :class:`.OperationInfo`:

  - make sure the **name** is consistant with the **op_name**
    of tensors in **output_tensors**
  - Just like :class:`.TensorInfo`, the ownership is established by passing
    a :class:`.uTensorGraph` as an argument to its constructor
  - make sure you update **n_inputs** and **n_outputs** when you make changes
    to **input_tensors** and **output_tensors**
  - When you try to transfer a op from one graph to the other,
    use :py:meth:`.OperationInfo.move_into`.
- About :class:`.uTensorGraph`:
  - please read the note list in :class:`.uTensorGraph`


Module members
--------------

.. autoapimodule:: utensor_cgen.ir.base
  :members:
  :exclude-members: topologic_order_graph, random_str, ConverterDispatcher, AttrValueConverter
