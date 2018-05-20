# Introduction

It's an end-to-end example of deploying a multilayer perceptron trained with `TensorFlow` to `mbed` device.

1. install `utensor_cgen`
    - you can find instructions for installation [here](https://github.com/uTensor/utensor_cgen#installation-python-2--3)
    - After installation, run `utensor-cli -v` to test the installation
2. open [`end_to_end.ipynb`](end_to_end.ipynb) with `jupyter notebook`
    - In this notebook, you will see how to training, serializing and quantizing a `TensorFlow` computational graph (`tf.Graph`) into a protobuf file
    - The graph protobuf file is saved as `simple_mnist.pb`. You can use it for the following demo.
    - this is how the graph looks like in `tensorboard`:<br/><br/><div><img src=readme_imgs/quant_mnist.png width=200 height=400 /> <img src=readme_imgs/quant_mnist_expend.png width=200 height=400 /></div>
3. run `utensor-cli simple_mnist.pb --output-nodes=y_pred`
    - It will create two directories, `constants` and `models`
    - In `models`, you should see two files, `simple_mnist.cpp` and `simple_mnist.hpp`.
    - In `constants` directory, you will find a subdirectory `simple_mnist` with many idx files.
        - copy `constants` directory to your SD card
4. Write a simple `main.cpp`
    - In the `main.cpp`, include `simple_mnist.hpp` file
    - The basic routine is as following:
        - Declare a `Context` object
        - Read input tensor(s) with `TensorIdxImporter` if any
        - Pass the `Context` object and the input tensor(s) to the function declared in the generated header file
        - Use `Context::get` to get the output tensor(s) you need
        - Make use of the output tensor(s)
    - See [`main.cpp`](main.cpp) for example
        - This `main.cpp` can be compiled and run on my [fork](https://github.com/dboyliao/mbed-simulator/tree/end2end)
        - It's a fork of Jan's amzing browser-based mbed simulator, [mbed-simulator](https://github.com/janjongboom/mbed-simulator)
