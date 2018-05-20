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

1. `$ pipenv install -d`
2. `$ pipenv shell`
    - this will spawn a subshell and activate the virtual environment for you
    - You should be able to use the cli now

**Note**: If you have trouble with installation with `pipenv`, try to remove `Pipfile.lock` first and run `pipenv install -d` again.

# Example

Please refer to [tests/deep_mlp](https://github.com/uTensor/utensor_cgen/tree/develop/tests/deep_mlp) for detailed example

# User Guild

Following steps are a general guild for user how to porting a `TensorFlow` protobuf file into a `uTensor` implementation:

1. install `utensor_cgent`
    - run `python3 setupt.py install`
2. run `utensor-cli graph.pb --output-nodes=NODE,NODE,...`
    - run `utensor-cli -h` for help
    - the `graph.pb` is the pb file of *original* graph (not quantized)

# Known Limitations

- If you want to use dropout with placeholders for the `keep_prob`, you have to name the `keep_prob` placeholder by any name that starts with "keep_prob".
    - You can still use any input tensor with name starts with "keep_prob" as long as it's not the output tensor of a placeholder node. 

# TODOs
1. (done) Freezed graph protobuff parser
2. (done)Tensor snippets for [`uTensor`](https://github.com/neil-tan/uTensor)
3. (done) Add template engine for richer flexibility
    - [jinja2](http://jinja.pocoo.org)
4. (done?) core code generator implementation
    - We need some refactoring, PRs are welcomed!
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
