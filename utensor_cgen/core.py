# -*- coding:utf8 -*-
import os
import idx2numpy as idx2np
from .pbparser import parse_pb
from .snippets import CreateTensorIdxSnippet, CreateTensorNewSnippet
from .snippets import CreateOpSnippet, register_template
from .composer import Composer
from ._snippets_base import SnippetContainer, Snippet
from ._types import TYPES_MAP

__all__ = ["CodeGenerator"]


class CodeGenerator(object):
  def __init__(self, pb_file: str, idx_dir):
    self.pb_file = pb_file
    self.idx_dir = idx_dir

  def generate(self, src_fname: str):
    """Generate source and header files
    """
    fname, _ = os.path.splitext(src_fname)
    header_fname = '{}.hpp'.format(fname)

    composer = Composer()
    container = SnippetContainer("get_ctx.cpp")
    container.add_header('"{}"'.format(header_fname))
    header_snippet = Snippet("get_ctx.hpp")
    print("Generate header file: {}".format(header_fname))
    with open(header_fname, "w") as wf:
      wf.write(header_snippet.render())

    print("Parsing {}".format(self.pb_file))
    graph_info, layers = parse_pb(self.pb_file)
    # TODO better snippet construction abstraction
    for layer in layers:
      for op_name in layer:
        op_type = graph_info[op_name]["op_type"]
        if op_type == 'Const':
          snippet = CreateTensorIdxSnippet()
          container.add_snippet(snippet)
        elif op_type == "Placeholder":
          pass
        elif op_type == "Add":
          pass
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
    print("Generate source file: {}".format(src_fname))
    with open(src_fname, "w") as wf:
      wf.write(composer.compose())

  def _save_data(self, tensor_name, array) -> str:
    pass

  def register_template(self, template_name, headers=None):
    register_template(template_name, headers)
