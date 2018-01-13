# -*- coding:utf8 -*-
import os

import numpy as np
import tensorflow as tf
import idx2numpy as idx2np

from .composer import Composer
from .operators import OperatorFactory
from .pbparser import parse_pb
from .optimizer import Optimizer
from .snippets import (CreateTensorIdxSnippet, CommentSnippet,
                       ContextHeaderSnippet, ContextSnippetsContainer)

__all__ = ["CodeGenerator"]


class CodeGenerator(object):
  def __init__(self, pb_file, idx_dir, embed_data_dir, debug_cmt=False):
    self.pb_file = pb_file
    if not os.path.exists(idx_dir):
      os.makedirs(idx_dir)
    self.idx_dir = idx_dir
    self.embed_data_dir = embed_data_dir.rstrip("/")
    self.debug_cmt = debug_cmt

  def generate(self, src_fname):
    """Generate source and header files
    """
    fname, _ = os.path.splitext(src_fname)
    graph_name, _ = os.path.splitext(os.path.basename(self.pb_file))
    header_snippet = ContextHeaderSnippet(fname, graph_name)

    composer = Composer()
    header_fname = '{}.hpp'.format(fname)
    container = ContextSnippetsContainer(graph_name, header_fname)

    opFactory = OperatorFactory()

    print("Parsing {}".format(self.pb_file))
    ops_info, ops_bfs, output_nodes = parse_pb(self.pb_file)
    construct_order = Optimizer.optimize(ops_info, ops_bfs, output_nodes)

    # TODO better snippet construction abstraction
    for op_id, (op_name, op_info, ref_counts, to_eval) in enumerate(construct_order, 1):
      op_type = op_info.op_type
      if op_type == "Placeholder":
        out_tname, _, _ = op_info.output_tensor[0]
        ref_count = ref_counts[0]
        container.template_vars["placeholders"].append(out_tname)
        container.template_vars["ref_counts"].append(ref_count)
        header_snippet.template_vars["placeholders"].append(out_tname)
      elif op_type == 'Const':
        out_tname, out_dtype, _ = op_info.output_tensor[0]
        pre_tname = self._prepare_tensor_name(out_tname)
        idx_fname = "{}.idx".format(pre_tname)
        snippet = CreateTensorIdxSnippet(self.embed_data_dir, out_tname,
                                         idx_fname=idx_fname,
                                         tf_dtype=out_dtype)
        container.add_snippet(snippet)
        idx_path = os.path.join(self.idx_dir, idx_fname)
        value = op_info.output_content[out_tname]
        self._save_data(idx_path, value, out_dtype)
      else:
        snippet = opFactory.createOperatorSnippet(op_info, ref_counts, to_eval)
        container.add_snippet(snippet)

      if self.debug_cmt:
        comments = ["<<< Operation id {}: {}".format(op_id, op_name),
                    ">>> Operation id {}: {}".format(op_id+1, op_name)]
        cmt_snippet = CommentSnippet(comments)
        container.add_snippet(cmt_snippet)
    composer.add_snippet(container)

    print("Generate header file: {}".format(header_fname))
    with open(header_fname, "w") as wf:
      wf.write(header_snippet.render())
    print("Generate source file: {}".format(src_fname))
    with open(src_fname, "w") as wf:
      wf.write(composer.compose())

  def _prepare_tensor_name(self, tensor_name):
    prepared = tensor_name.replace(":", "_").replace("/", "_")
    return prepared

  def _save_data(self, path, value, tf_dtype):
    if tf_dtype in [tf.uint8, tf.qint8, tf.quint8]:
      np_dtype = np.uint8
    elif tf_dtype in [tf.int32, tf.qint32]:
      np_dtype = np.int32
    else:
      np_dtype = np.float32

    if value.shape == ():
      value = np.array([value], dtype=np_dtype)
    else:
      value = value.astype(np_dtype)
    with open(path, "wb") as fid:
      idx2np.convert_to_file(fid, value)
    print("saving {}".format(path))
