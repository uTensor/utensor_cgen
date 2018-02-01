# Installation (Python 2 & 3)

- installation with `setup.py`
```
python setup.py install
```

- installation with `pip`
```
pip install utensor_cgen
```

# Develop Environment

We use `pipenv` to setup the develop environment.

You can go to this [repo](https://github.com/pypa/pipenv) for detail information about `pipenv`.

## Setup with `pipenv`

1. `# pipenv install -d`
2. `# pipenv shell`
    - this will spawn a subshell and activate the virtual environment for you
    - You should be able to use the cli now

# Example

Please refer to [tests/deep_mlp](https://github.com/uTensor/utensor_cgen/tree/develop/tests/deep_mlp) for detailed example

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
6. (done) `MANIFAST.in` for the `setup.py`
7. Relation among snippets/containers
    - shared template variables? (headers, shared placeholders...etc)
8. Better configuration schema
    - json
    - yaml
    - or ?
