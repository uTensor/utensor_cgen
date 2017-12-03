# -*- coding:utf8 -*-
import os
import numpy as np
import idx2numpy as idx2np
import tensorflow as tf
from .pbparser import parse_pb
from .snippets import *
from .snippets import register_template
from .composer import Composer
from ._snippets_base import SnippetContainer, Snippet
from ._types import TF_TYPES_MAP

__all__ = ["CodeGenerator"]


class CodeGenerator(object):
  def __init__(self, pb_file: str, idx_dir: str, embed_data_dir: str):
    self.pb_file = pb_file
    if not os.path.exists(idx_dir):
      os.makedirs(idx_dir)
    self.idx_dir = idx_dir
    self.embed_data_dir = embed_data_dir.rstrip("/")

  def generate(self, src_fname: str):
    """Generate source and header files
    """
    fname, _ = os.path.splitext(src_fname)
    header_fname = '{}.hpp'.format(fname)

    composer = Composer()
    container = SnippetContainer("get_ctx.cpp")
    container.add_header('"{}"'.format(header_fname))

    print("Parsing {}".format(self.pb_file))
    graph_info, layers = parse_pb(self.pb_file)

    # TODO better snippet construction abstraction
    for layer in layers:
      for op_name in layer:
        op_info = graph_info[op_name]
        op_type = op_info["op_type"]
        if op_type == "Placeholder":
          # TODO what is a placeholder in uTensor?
          pass
        elif op_type == 'Const':
          for out_tname, out_dtype in op_info["output_tensor"]:
            pre_tname = self._prepare_tensor_name(out_tname)
            idx_fname = "{}.idx".format(pre_tname)
            snippet = CreateTensorIdxSnippet(self.embed_data_dir, out_tname,
                                             idx_fname=idx_fname,
                                             tf_dtype=out_dtype)
            container.add_snippet(snippet)
            idx_path = os.path.join(self.idx_dir, idx_fname)
            value = op_info["output_content"][out_tname]
            self._save_data(idx_path, value, out_dtype)
        elif op_type == "Add":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          output, _ = op_info["output_tensor"][0]
          tf_dtype = op_info["input_tensor"][0][1]
          snippet = AddOpSnippet(inputs, output, tf_dtype=tf_dtype)
          container.add_snippet(snippet)
        elif op_type == "ArgMax":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          output, out_dtype = op_info["output_tensor"][0]
          _, in_dtype = op_info["input_tensor"][0]
          snippet = ArgMaxOpSnippet(inputs, output, in_dtype, out_dtype)
          container.add_snippet(snippet)
        elif op_type == "Dequantize":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          output, _ = op_info["output_tensor"][0]
          snippet = DequantizeOpSnippet(inputs, output)
          container.add_snippet(snippet)
        elif op_type == "Max":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          output, _ = op_info["output_tensor"][0]
          snippet = MaxOpSnippet(inputs, output)
          container.add_snippet(snippet)
        elif op_type == "Min":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          output, _ = op_info["output_tensor"][0]
          snippet = MinOpSnippet(inputs, output)
          container.add_snippet(snippet)
        elif op_type == "QuantizeV2":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          outputs = [tname for tname, _ in op_info["output_tensor"]]
          snippet = QuantizeV2OpSnippet(inputs, outputs)
          container.add_snippet(snippet)
        elif op_type == "QuantizedMatMul":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          outputs = [tname for tname, _ in op_info["output_tensor"]]
          x_dtype = op_info["input_tensor"][0][1]
          w_dtype = op_info["input_tensor"][1][1]
          out_dtype = op_info["output_tensor"][0][1]
          snippet = QuantizedMatMulOpSnippet(inputs, outputs, x_dtype, w_dtype, out_dtype)
          container.add_snippet(snippet)
        elif op_type == "QuantizedRelu":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          outputs = [tname for tname, _ in op_info["output_tensor"]]
          _, in_dtype = op_info["input_tensor"][0]
          _, qout_dtype = op_info["output_tensor"][0]
          _, out_dtype = op_info["output_tensor"][1]
          snippet = QuantizedReluOpSnippet(inputs, outputs, in_dtype, out_dtype, qout_dtype)
          container.add_snippet(snippet)
        elif op_type == "RequantizationRange":
          pass
        elif op_type == "Requantize":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          outputs = [tname for tname, _ in op_info["output_tensor"]]
          snippet = RequantizeOpSnippet(inputs, outputs)
          container.add_snippet(snippet)
        elif op_type == "Reshape":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          output, _ = op_info["output_tensor"][0]
          snippet = ReshapeOpSnippet(inputs, output)
          container.add_snippet(snippet)
        else:
          raise ValueError("unsupported op type in uTensor: {}".format(op_type))
    composer.add_snippet(container)

    header_snippet = Snippet("get_ctx.hpp")
    print("Generate header file: {}".format(header_fname))
    with open(header_fname, "w") as wf:
      wf.write(header_snippet.render())
    print("Generate source file: {}".format(src_fname))
    with open(src_fname, "w") as wf:
      wf.write(composer.compose())

  def _prepare_tensor_name(self, tensor_name: str) -> str:
    prepared = tensor_name.replace(":", "_").replace("/", "_")
    return prepared

  def _save_data(self, path, value, tf_dtype):
    if tf_dtype in [tf.uint8, tf.qint8]:
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
