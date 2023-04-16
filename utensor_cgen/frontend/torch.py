"""
TODO: torch script support (https://pytorch.org/docs/stable/jit.html)
"""
from io import BytesIO

import torch
import torch.nn
import torch.onnx

from utensor_cgen.frontend import FrontendSelector
from utensor_cgen.frontend.base import Parser

from .onnx import OnnxParser as _OnnxParser


@FrontendSelector.register(target_exts=[".torch"])
class TorchModuleParser(Parser):
  def __init__(self, config):
    self._onnx_parser = _OnnxParser(config)

  def parse(self, model_file: str, output_nodes=None, model_name=None, **kwargs):
    inputs_file = kwargs.get("inputs_file", None)
    if inputs_file is None:
      raise ValueError("--inputs-file is required for torch model file")
    input_args = torch.load(inputs_file)
    model: torch.nn.Module = torch.load(model_file)
    buffer = BytesIO()
    torch.onnx.export(model, input_args, buffer)
    buffer.seek(0)
    return self._onnx_parser.parse(
        buffer,
        output_nodes=output_nodes,
        model_name=model_name,
    )
