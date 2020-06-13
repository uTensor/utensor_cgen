# Adding custom operators

This tutorial shows how to register custom operators in the uTensor code generation tools using the plugin system. We will start by defining a custom Layer in Keras using existing Tensorflow operators for clarity and brevity, however extending the concepts in this tutorial to map fully custom framework operators to the uTensor form should be straightforward.  

## The operator lifecycle
The operator lifecycle loosely falls into 3 categories, the Frontend, the Transformer, and the Backend. 

- The goal of the `Frontend` is to lift tensor and operator graphs from the various frameworks (Tensorflow, ONNX, etc.) into the uTensor Intermediate Representation (IR). At this stage, the IR makes no assumptions on whether or not a particular operator is "supported", so for the sake of this tutorial is therefore largely ignored. Finally, at the end of this stage and right before the next is the `Legalizer`. This legalization simply ensures that the IR can be processed in the most generic form. For example, we might simply rename `Dense` or `DenseLayer` to `FullyConnectedOperator`.
- The goal of the `Transformer`, as its name implies, is to transform uTensor IR graphs. For example, generating memory plans, rewriting graphs, and generally optimizing inference. Again, this is predominantly operator agnostic so is mostly ignored in this tutorial. 
- The goal of the `Backend` is to do the final lowering of an optimized IR into either code or binary forms. At the end of the transformation pipeline, the final graph is then *lowered* into a target `Backend`. This lowering process is Backend specific and allows the backend to inject additional attributes into the IR before finalizing a strategy for mapping IR components to their respective backend handlers. For example, in the uTensor backened we can inject namespace information to prioritize CMSIS handlers over reference floating point depending on if the operator in question is quantized. Next, the uTensor backend can use these respective handlers to compose a set of code snippets, which ultimately becomes the output model code.


This means there are a total of 5 locations where we might want to register our custom operator, but some of them may be optional based on use-case:

1. Frontend parsing
2. Legalization
3. Backend Lowering
4. Backend Component
5. Backend Snippet


## The Model Graph

![reduce-model](images/reduceModel.svg)

```bash
$ utensor-cli --plugin custom_operators convert tf_models/reduceModel.tflite
```
