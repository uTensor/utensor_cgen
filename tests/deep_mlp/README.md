# Introduction

It's an end-to-end example of deploying a multilayer perceptron trained with `TensorFlow` to `mbed` device.

1. install `utensor_cgen`
    - you can find instructions for installation [here](https://github.com/uTensor/utensor_cgen#installation-python-2--3)
    - After installation, run `utensor-cli -v` to test the installation
2. open `end_to_end.ipynb` with `jupyter notebook`
    - In this notebook, you will see how to training, serializing and quantizing a `TensorFlow` computational graph (`tf.Graph`) into a protobuf file
    - The quantized graph protobuf file is saved as `qunat_mnist.pb`. You can use it for the following demo.
    - this is how the graph looks like in `tensorboard`:<br/><br/><div><img src=readme_imgs/quant_mnist.png width=150 height=250 /> <img src=readme_imgs/quant_mnist_expend.png width=200 height=400 /></div>
3. hello



Testing cpp codes for mbed simulator

https://github.com/janjongboom/mbed-simulator
