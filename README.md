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

# TODOs
1. (done) Freezed graph protobuff parser
2. (done)Tensor snippets for [`uTensor`](https://github.com/neil-tan/uTensor)
3. (done) Add template engine for richer flexibility
    - [jinja2](http://jinja.pocoo.org)
4. core code generator implementation
    - What is a `Placeholder` in uTensor?
5. type alias in C/C++
    - ex: use `uint8_t` or `unsigned char`?
    - a lot more about this.... 
6. `MANIFAST.in` for the `setup.py`
7. Relation among snippets/containers
    - shared template variables?
8. Better configuration schema
    - json
    - yaml
    - or ?
