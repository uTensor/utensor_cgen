# -*- coding: utf8 -*-
import os
import re

import numpy as np
import idx2numpy as idx2np
import tensorflow as tf

__all__ = ["save_idx", "save_consts", "save_graph"]


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
