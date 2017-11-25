# -*- coding:utf8 -*-
import io
import sys
import tensorflow as tf
from tensorflow.python.framework import graph_util  # pylint: disable=E0611

__all__ = ["parse_pb"]

__KNOWN_OPS = {}  # Add known uTensor ops <--> tensorflow ops mapping


def parse_pb(file_or_path, output_nodes=None):
    """
    arguments
    =========
    - file_or_path: a file object or a path string of the pb file
    - output_nodes: list of output node names

    returns
    =======
    - nodes: mapping from node name to its NodeDef object
    """
    if sys.version_info.major < 3:
        file_type = (file, io.IOBase)  # pylint: disable=E0602
    else:
        file_type = io.IOBase

    if isinstance(file_or_path, str):
        fid = open(file_or_path, "rb")
    elif isinstance(file_or_path, file_type):
        fid = file_or_path
    else:
        raise ValueError("`file_or_path` has to be either file object or path string")

    # load pb file
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fid.read())
    fid.close()

    if output_nodes:
        sub_graph_def = graph_util.extract_sub_graph(graph_def, output_nodes)
    else:
        sub_graph_def = graph_def

    return dict((node.op, node) for node in sub_graph_def.node)  # pylint: disable=E1101
