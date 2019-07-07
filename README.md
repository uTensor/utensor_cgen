# Installation (Python 2 & 3)

## For Users

- with `setup.py`
```
python setup.py install
```

- with `pip`
```
pip install utensor_cgen
```

## For Developers:

- with `pip`
```
pip install -e .[dev]
```

- with `pipenv`
    1. `$ PIPENV_VENV_IN_PROJECT=1 pipenv install -d`
    2. `$ pipenv shell`
        - this will spawn a subshell and activate the virtual environment for you
        - You should be able to use the cli now  

You can go to this [repo](https://github.com/pypa/pipenv) for detail information about `pipenv`.

### Troubleshooting with `pipenv`

- If you have trouble with installation using `pipenv`, try `PIPENV_VENV_IN_PROJECT=1 pipenv install -d --skip-lock`
- there is known issue of `pip` and `pipenv`, plz refer to this [issue](https://github.com/pypa/pipenv/issues/2924) for detail
    - short answer: downgrade to `pip==18.0` may help :)
- Tensorflow requires `setuptools<=39.1.0` (the latest is `40.4.3` by the time this README is writen)
    - plz downgrade to `setuptools==39.1.0`
    - my recommendation is to use `virtualenv`

# Overall Architecture

```
  ============       +-----------------+       ===================
|| model file || --> | frontend Parser | --> || uTensorGraph (IR) || 
  ============       +-----------------+       ===================
                                                           |
                 +-------------------------------+         |
                 |       graph transformer       |         |
                 | (legalization & optimization) | <------‘ 
                 +-------------------------------+
                                |
                                v
                     ===========================
                   ||       uTensorGraph        ||
                   || (legalized and optimized) ||
                     ===========================
                                   |
+--------------------------+       |
| backend (code generator) | <----‘  
+--------------------------+
     |
     `---> (target files, ex: model.cpp, model.hpp, weights.idx)
```

# Basic Usage

## `utensor-cli show <model.pb>`

Show all nodes and detailed information of given pb file.

Run `utensor-cli show --help` for detailed information.

## `utensor-cli convert --output-nodes=<node_name>[,<node_name>,...] <model.pb>`

Convert given pb file into cpp/hpp files.

Note that `--output-nodes` is required options. It's the names of nodes you want to output, seperated by comma if there are many.

In graph theory terminology, they are `leaf` nodes of your graph.

Run `utensor-cli convert --help` for detailed information.

# Use `utensor_cgen` as Library

## Subgraph Isomorphic Matcher and Nodes Fusion

With `uTensorGraphMatcher`, performing common subgraph tasks such as isomorphic matching along with replacing or manipulating the matched subgraph(s) takes just a few line of code!

- (**Future Work**) High-level api for graph pattern declaration. Currently relies on `TensorFlow` api for building graph and converting to `utensor_cgen` IR graph.

### Node Fusion

Note: we'll use operation/node/layer interchangeably in the documentation

- It's commonly seen pattern in convolution neural network (`CNN`), `conv -> relu -> pooling`. That is, a 2D convolution followed by a relu layer and then a pooling down sampling layer.
- With our `uTensorGraphMatcher`, you can locate such pattern in your `CNN` model and fuse/replace matched nodes into one optimized `QuantizedFusedConv2DMaxpool` node.
    - Left: original graph
    - Middle: matched convolution layer
    - Right: replace the matched layer with specialized `QuantizedFusedConv2DMaxpool` node

![conv-pool-fuce](images/conv_pool_fuse.png)

### Dropout Layer Removal

- Though `dropout` is an effective technique to improve training performance of your model, it's not necessary during inference phrase.
- In the mainstream frameworks such as `Tensorflow` or `PyTorch`, an `dropout` layer is typically implemented with other elementary operations/nodes. As a result, finding and removing those nodes for interence optimization (both in model size and prediciton time) is not trivial and error prone.
- With our `uTensorGraphMatcher`, you can find and remove the dropout nodes as illustrated in the following picture.
    - Left: original graph with dropout Layers
    - Middle: matched dropout layers
    - Right: graph with dropout layers removed
![cnn-dropout](images/cnn_dropout.png)

# Examples

- [Deep Multilayer Perceptron](https://github.com/uTensor/utensor_cgen/tree/develop/tests/deep_mlp)
- [End-to-End Convolution NN](https://github.com/uTensor/simple_cnn_tutorial)


# User Guide

Following steps are a general guild for user how to porting a `TensorFlow` protobuf file into a `uTensor` implementation:

1. install `utensor_cgent`
    - run `python3 setupt.py install`
2. run `utensor-cli convert --output-nodes='NODE,NODE,...' graph.pb`
    - run `utensor-cli -h` for help
    - the `graph.pb` is the pb file of *original* graph (not quantized)
3. If you want to see what ops/nodes are in the pb file, you can run `utensor-cli show <pbfile>`

# How to test (for Developer)

1. follow the steps in [setup](#setup-with-pipenv) section
2. run `make tests`
    - Or you can use `pipenv run pytest tests` instead

# Philosophy

- [12 Factor CLI App](https://medium.com/@jdxcode/12-factor-cli-apps-dd3c227a0e46?fbclid=IwAR1Gfq0D7oh3b-mXaIMV3RwYu39TAPrPXfz5sBKC4Rz1t-cckvC8WjBVl_w)

# TODOs
1. (done?) core code generator implementation
    - We need some refactoring, PRs are welcomed!
2. type alias in C/C++
    - ex: use `uint8_t` or `unsigned char`?
    - a lot more about this.... 
3. Relation among snippets/containers
    - shared template variables? (headers, shared placeholders...etc)
4. Better configuration schema
    - json
    - yaml
    - or ?
