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
pip instal -e .[dev]
```

- with `pipenv`
    1. `$ PIPENV_VENV_IN_PROJECT=1 pipenv install -d`
    2. `$ pipenv shell`
        - this will spawn a subshell and activate the virtual environment for you
        - You should be able to use the cli now  

You can go to this [repo](https://github.com/pypa/pipenv) for detail information about `pipenv`.

**Note**: If you have trouble with installation using `pipenv`, try `PIPENV_VENV_IN_PROJECT=1 pipenv install -d --skip-lock`

### If All Fails...

1. You can use `docker`
    - run `docker pull dboyliao/utensor-cli` for a pre-build docker image
    - or run `docker build -t <user_name>/utensor-cli .` to build the docker image
      by yourself, where `<user_name>` can be any user name you want.
2. Run the docker image
    - `docker run -it dboyliao/utensor-cli`
    - or `docker run -it <user_name>/utensor-cli` if you want to use the image you just
      build.

Please refer to `docker` [documentation](https://docs.docker.com/get-started/) for detail.

# Example

Please refer to [tests/deep_mlp](https://github.com/uTensor/utensor_cgen/tree/develop/tests/deep_mlp) for detailed example

# User Guide

Following steps are a general guild for user how to porting a `TensorFlow` protobuf file into a `uTensor` implementation:

1. install `utensor_cgent`
    - run `python3 setupt.py install`
2. run `utensor-cli graph.pb --output-nodes=NODE,NODE,...`
    - run `utensor-cli -h` for help
    - the `graph.pb` is the pb file of *original* graph (not quantized)

# How to test (for Developer)

1. follow the steps in [setup](#setup-with-pipenv) section
2. run `make tests`
    - Or you can use `pipenv run pytest tests` instead

# Known Limitations

- If you want to use dropout with placeholders for the `keep_prob`, you have to name the `keep_prob` placeholder by any name that starts with "keep_prob".
    - You can still use any input tensor with name starts with "keep_prob" as long as it's not the output tensor of a placeholder node.
    - You can't wrap `dropout` in any `namescope` 

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
