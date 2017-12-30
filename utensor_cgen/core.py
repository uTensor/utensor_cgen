# -*- coding:utf8 -*-
import os
import numpy as np
import idx2numpy as idx2np
import tensorflow as tf
from .pbparser import parse_pb
from .snippets import *  # pylint: disable=W0401
from .snippets import register_template
from .composer import Composer
from ._snippets_base import SnippetContainer, Snippet
from ._types import TF_TYPES_MAP
from .operators import OperatorFactory

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
    header_fname = '{}.hpp'.format(fname)
    header_snippet = Snippet("get_ctx.hpp")
    header_snippet.template_vars["header_guard"] = "_{}_H".format(fname.upper())
    header_snippet.template_vars["graph_name"] = graph_name
    header_snippet.template_vars["placeholders"] = []

    composer = Composer()
    container = SnippetContainer("get_ctx.cpp")
    container.template_vars["graph_name"] = graph_name
    container.template_vars["placeholders"] = []
    container.add_header('"{}"'.format(header_fname))

    opFactory = OperatorFactory()

    print("Parsing {}".format(self.pb_file))
    graph_info, layers = parse_pb(self.pb_file)

    # TODO better snippet construction abstraction
    for layer_id, layer in enumerate(layers, 1):
      for op_name in layer:
        op_info = graph_info[op_name]
        op_type = op_info["op_type"]
        if op_type == "Placeholder":
          out_tname, _, _ = op_info["output_tensor"][0]
          container.template_vars["placeholders"].append(out_tname)
          header_snippet.template_vars["placeholders"].append(out_tname)
        elif op_type == 'Const':
          for out_tname, out_dtype, _ in op_info["output_tensor"]:
            pre_tname = self._prepare_tensor_name(out_tname)
            idx_fname = "{}.idx".format(pre_tname)
            snippet = CreateTensorIdxSnippet(self.embed_data_dir, out_tname,
                                             idx_fname=idx_fname,
                                             tf_dtype=out_dtype)
            container.add_snippet(snippet)
            idx_path = os.path.join(self.idx_dir, idx_fname)
            value = op_info["output_content"][out_tname]
            self._save_data(idx_path, value, out_dtype)
        else:
          snippet = opFactory.createOperatorSnippet(op_info)  
          container.add_snippet(snippet)

      if self.debug_cmt:
        comments = ["<<< Graph Layer {}".format(layer_id), 
                    ">>> Graph Layer {}".format(layer_id+1)]
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

  def register_template(self, template_name, headers=None):
    register_template(template_name, headers)
