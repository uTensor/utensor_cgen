import numpy as np

import tensorflow as tf
from utensor_cgen.frontend.tensorflow import GraphDefParser
from utensor_cgen.matcher import uTensorGraphMatcher
from utensor_cgen.utils import prune_graph, topologic_order_graph
from utensor_cgen.transformer import TFLiteExporter

def test_tflite_fb_write(sample_ugraph):
    exporter = TFLiteExporter()
    ugraph = exporter.transform(sample_ugraph)
    exporter.output()
    print(exporter.output())

    test_pass = True
    assert test_pass, 'error message here'
