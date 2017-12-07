# Tests for `utensor_cgen`

## How to test

1. set working directory as `tests`
2. run `make`
    - it will make a `cpp` directory, which contains all generated idx, source code and header files, in each subdirectories. (ex: `add`, `argmax`,...etc)
    - compile those file with your main program and run test on the develop board
3. The testing directories typically contain two pb files, such as `test_add.pb` and `test_quant_add.pb`. The former is the original graph and the later is the quanitized graph of the original graph.
4. There should be also one or multiple idx files with names start with `output`, such as `output_z.idx`. Those are the output value of the graph. You can use them to test output on the develop board.

## TODOs

- Test for deep multiperceptron models
- Test for CNN

## References

- [Quantization - TensorFlow](https://www.tensorflow.org/performance/quantization)
    - [python script](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/quantization/quantize_graph.py)
