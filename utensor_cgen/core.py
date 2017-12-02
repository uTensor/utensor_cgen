# -*- coding:utf8 -*-
import os
import numpy as np
import idx2numpy as idx2np
from .pbparser import parse_pb
from .snippets import CreateTensorIdxSnippet, CreateTensorNewSnippet
from .snippets import AddOpSnippet, register_template
from .composer import Composer
from ._snippets_base import SnippetContainer, Snippet
from ._types import TYPES_MAP

__all__ = ["CodeGenerator"]


class CodeGenerator(object):
  def __init__(self, pb_file: str, idx_dir: str):
    self.pb_file = pb_file
    if not os.path.exists(idx_dir):
      os.makedirs(idx_dir)
    self.idx_dir = idx_dir

  def generate(self, src_fname: str, mbed_data_dir: str):
    """Generate source and header files
    """
    if mbed_data_dir.endswith("/"):
      mbed_data_dir = mbed_data_dir[:-1]
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
        if op_type == 'Const':
          for out_tname, out_dtype in op_info["output_tensor"]:
            pre_tname = self._prepare_tensor_name(out_tname)
            idx_fname = "{}.idx".format(pre_tname)
            snippet = CreateTensorIdxSnippet(mbed_data_dir, out_tname,
                                             idx_fname=idx_fname,
                                             tf_dtype=out_dtype)
            container.add_snippet(snippet)
            idx_path = os.path.join(self.idx_dir, idx_fname)
            value = op_info["output_content"][out_tname]
            self._save_data(idx_path, value)
        elif op_type == "Placeholder":
          pass
        elif op_type == "Add":
          inputs = [tname for tname, _ in op_info["input_tensor"]]
          output, _ = op_info["output_tensor"][0]
          tf_dtype = op_info["input_tensor"][0][1]
          snippet = AddOpSnippet(inputs, output, tf_dtype=tf_dtype)
          container.add_snippet(snippet)
        elif op_type == "ArgMax":
          pass
        elif op_type == "Dequantize":
          pass
        elif op_type == "Max":
          pass
        elif op_type == "Min":
          pass
        elif op_type == "QuantizeV2":
          pass
        elif op_type == "QuantizedMatMul":
          pass
        elif op_type == "QuantizedRelu":
          pass
        elif op_type == "RequantizationRange":
          pass
        elif op_type == "Requantize":
          pass
        elif op_type == "Reshape":
          pass
        else:
          pass
          # raise ValueError("unsupported op type in uTensor")
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

  def _save_data(self, path, arr):
    if arr.shape == ():
      arr = np.array([arr])
    with open(path, "wb") as fid:
      idx2np.convert_to_file(fid, arr)
    print("saving {}".format(path))

  def register_template(self, template_name, headers=None):
    register_template(template_name, headers)
