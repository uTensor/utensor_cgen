# -*- coding: utf8 -*-
import os
import re
from copy import deepcopy

import numpy as np
import idx2numpy as idx2np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.tools.graph_transforms import TransformGraph

from utensor_cgen.logger import logger

__all__ = ["save_idx", "save_consts", "save_graph", "log_graph", "KWArgsParser"]


def log_graph(graph_or_graph_def, logdir):
  if isinstance(graph_or_graph_def, tf.GraphDef):
    graph = tf.Graph()
    with graph.as_default():
      tf.import_graph_def(graph_or_graph_def, name='')
  else:
    graph = graph_or_graph_def
  tf.summary.FileWriter(logdir, graph=graph).close()


def save_idx(arr, fname):
  if arr.shape == ():
    arr = np.array([arr], dtype=arr.dtype)
  if arr.dtype in [np.int64]:
    logger.warning("unsupported int format for idx detected: %s, using int32 instead", arr.dtype)
    arr = arr.astype(np.int32)
  out_dir = os.path.dirname(fname)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir)
  with open(fname, "wb") as fid:
    idx2np.convert_to_file(fid, arr)
  logger.info("%s saved", fname)


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
  logger.info("%s saved", graph_fname)


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

def _sanitize_op_name(op_name):
  """
  Sanitize the op name
  - ignore '^' character of control input
  """
  if op_name.startswith('^'):
    return op_name[1:]
  return op_name

def parse_tensor_name(tname):
  """Adapt from TensorFlow source code
  tensor name --> (op_name, index)
  """
  components = tname.split(":")
  if len(components) == 2:
    op_name = _sanitize_op_name(components[0])
    try:
      output_index = int(components[1])
    except ValueError:
      raise ValueError("invalid output index: {}".format(tname))
    return (op_name, output_index)
  elif len(components) == 1:
    op_name = _sanitize_op_name(components[0])
    return (op_name, 0)
  else:
    raise ValueError("invalid tensor name: {}".format(tname))

class NamescopedKWArgsParser:

  def __init__(self, name_space, kwargs):
    ns_pattern = re.compile(r'^{}__([^\d\W][\w\d_]*)'.format(name_space))
    self._namespace = name_space
    self._private_kwargs = {}
    self._shared_kwargs = {}
    for key, value in kwargs.items():
      match = ns_pattern.match(key)
      if match:
        argname = match.group(1)
        self._private_kwargs[argname] = value
      else:
        self._shared_kwargs[key] = value
  
  def get(self, argname, default=None):
    try:
      return self._private_kwargs[argname]
    except KeyError:
      return self._shared_kwargs.get(argname, default)
  
  def as_dict(self):
    kwargs = deepcopy(self._private_kwargs)
    for k, v in self._shared_kwargs.items():
      if not k in kwargs:
        kwargs[k] = v
    return kwargs

  def __repr__(self):
    d = dict(('%s__%s' % (self._namespace, k), v)
              for k, v in self._private_kwargs.items())
    repr_str = ('KWArgsParser(' + 
                '%s, ' % self._namespace + 
                '%s)' % d)
    return repr_str

  def __getitem__(self, argname):
    try:
      return self._private_kwargs[argname]
    except KeyError:
      return self._shared_kwargs[argname]
