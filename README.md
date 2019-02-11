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

# Example

Please refer to [tests/deep_mlp](https://github.com/uTensor/utensor_cgen/tree/develop/tests/deep_mlp) for detailed example

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

# Known Limitations

- If you want to use dropout with placeholders for the `keep_prob`, you have to name the `keep_prob` placeholder by any name that starts with "keep_prob".
    - You can still use any input tensor with name starts with "keep_prob" as long as it's not the output tensor of a placeholder node.
    - You can't wrap `dropout` in any `namescope` 

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
