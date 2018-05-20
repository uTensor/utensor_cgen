# -*- coding: utf8 -*-
import os
import re

import numpy as np
import idx2numpy as idx2np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph

from .parser.utils import log_graph

__all__ = ["save_idx", "save_consts", "save_graph", "log_graph"]


def save_idx(arr, fname):
  if arr.shape == ():
    arr = np.array([arr], dtype=arr.dtype)
  if arr.dtype in [np.int64]:
    print("unsupported int format for idx detected: {}, using int32 instead".format(arr.dtype))
    arr = arr.astype(np.int32)
  out_dir = os.path.dirname(fname)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir)
  with open(fname, "wb") as fid:
    idx2np.convert_to_file(fid, arr)
  print("{} saved".format(fname))


def save_consts(sess, out_dir="."):
  out_dir = os.path.expanduser(out_dir)
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)
  graph = sess.graph
  graph_def = sess.graph.as_graph_def()
  for node in graph_def.node:
    if node.op == "Const":
      op = graph.get_operation_by_name(node.name)
      for out_tensor in op.outputs:
        arr = out_tensor.eval()
        tname = re.sub(r'[/:]', '_', out_tensor.name)
        idx_fname = os.path.join(out_dir, "{}.idx".format(tname))
        save_idx(arr, idx_fname)


def save_graph(graph, graph_name="graph", out_dir="."):
  out_dir = os.path.expanduser(out_dir)
  graph_fname = os.path.join(out_dir, "{}.pb".format(graph_name))
  with tf.gfile.FastGFile(graph_fname, "wb") as fid:
    fid.write(graph.as_graph_def().SerializeToString())
  print("{} saved".format(graph_fname))


def prepare_meta_graph(meta_graph_path, output_nodes, chkp_path=None):
  """
  Cleanup and freeze the graph from meta graph

  1. remove training nodes
  2. convert variable to constants
  """
  graph = tf.Graph()
  saver = tf.train.import_meta_graph(meta_graph_path,
                                     clear_devices=True,
                                     graph=graph)
  if chkp_path is None:
    chkp_path = meta_graph_path.replace(".meta", "")
  with tf.Session(graph=graph) as sess:
    saver.restore(sess, chkp_path)
    graph_def = graph_util.remove_training_nodes(sess.graph_def)
    sub_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                              input_graph_def=graph_def,
                                                              output_node_names=output_nodes)
  return sub_graph_def
