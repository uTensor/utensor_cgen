# Installation (Python3)

## `setup.py`

run `python3 setup.py install`

## `pip`

run `pip install utensor_cgen`

This package is under beta development, using `virtualenv` is recommanded.

# Example
1. `example.py`:
    1. run `python3 example.py` and it should generate a `main.cpp` file.
    2. compile it and run, you should see familier hello world message
2. `simple_graph.pb`:
    1. install `utensor_cgen` by running `python3 setup.py install`
    2. run `python3 -m utensor_cgen simple_graph.pb`
    3. it will save constant tensor data in `idx_data/` and generate two files, `model.hpp` and `model.cpp`.
    4. compile your `uTensor` project with `model.hpp` and `model.cpp` and copy `idx_data/` to your SD card.
    5. You should have a running simple graph.

<center>
<img alt=simple-graph src=images/simple_graph.png />
</center>

# User Guild

Following steps are a general guild for user how to porting a `TensorFlow` protobuf file into a `uTensor` implementation:

1. Freeze and quantize your graph
    - [Freezing](https://www.tensorflow.org/extend/tool_developers/#freezing)
    - [Quantization](https://www.tensorflow.org/performance/quantization)
        - An alternative is to use the [`quantize_graph.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/quantization/quantize_graph.py) script
        - it should output one qunatized pb file, say `quantized_graph.pb`
2. install `utensor_cgent`
    - run `python3 setupt.py install`
3. run `utensor-cli quantized_graph.pb`, where `quantized_graph.pb` is the output pb file you get from step **1**
    - run `utensor-cli -h` for help

# TODOs
1. (done) Freezed graph protobuff parser
2. (done)Tensor snippets for [`uTensor`](https://github.com/neil-tan/uTensor)
3. (done) Add template engine for richer flexibility
    - [jinja2](http://jinja.pocoo.org)
4. (done?) core code generator implementation
    - We need some refatoring, PRs are welcomed!
5. type alias in C/C++
    - ex: use `uint8_t` or `unsigned char`?
    - a lot more about this.... 
6. `MANIFAST.in` for the `setup.py`
7. Relation among snippets/containers
    - shared template variables? (headers, shared placeholders...etc)
8. Better configuration schema
    - json
    - yaml
    - or ?
