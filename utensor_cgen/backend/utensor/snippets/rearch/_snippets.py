from utensor_cgen.backend.utensor.snippets._types import UTENSOR_TYPES_MAP, NP_TYPES_MAP
from utensor_cgen.backend.utensor.snippets._base import Snippet, SnippetBase

__all__ = ['RomTensorSnippet', 'DeclareOpSnippet', 'AddOpEvalSnippet', 'SimpleContainer']

class _SnippetBase(Snippet):
  __headers__ = set(['"uTensor/uTensor.hpp"'])


class RomTensorSnippet(_SnippetBase):
  __template_name__ = 'snippets/rearch/declare_rom_tensor.cpp'

  def __init__(self, tensor_var_name, buffer_var_name, tensor):
    Snippet.__init__(self)
    shape = tensor.shape or [1]
    self.template_vars = {
      'tensor_var_name': tensor_var_name,
      'shape': shape,
      'buffer_var_name': buffer_var_name,
      'utensor_dtype': UTENSOR_TYPES_MAP[tensor.dtype]
    }


class DeclareOpSnippet(_SnippetBase):
  __template_name__ = 'snippets/rearch/declare_op.cpp'

  def __init__(self, op_type, dtypes, op_var_name):
    Snippet.__init__(self)
    self.template_vars['op_type'] = op_type
    self.template_vars['dtypes'] = dtypes
    self.template_vars['op_var_name'] = op_var_name


class OpEvalSnippet(_SnippetBase):
  __template_name__ = 'snippets/rearch/eval_op.cpp'
  __inputs__ = []
  __outputs__ = []
  

  def __init__(self, op_info, op_name, tensor_var_map, dtypes):
    Snippet.__init__(self)
    input_map = {
      name: tensor_var_map[tensor.name]
      for name, tensor in zip(self.__inputs__, op_info.input_tensors)
    }
    output_map = {
      name: tensor_var_map[tensor.name]
      for name, tensor in zip(self.__outputs__, op_info.output_tensors)
    }
    out_shapes_map = {
      tensor_var_map[tensor.name]: tensor.shape or [1]
      for tensor in op_info.output_tensors
    }
    out_dtypes_map = {
      tensor_var_map[tensor.name]: UTENSOR_TYPES_MAP[tensor.dtype]
      for tensor in op_info.output_tensors
    }
    utensor_dtypes = [NP_TYPES_MAP[dtype].tensor_type_str for dtype in dtypes]
    if utensor_dtypes:
      op_type = '{}<{}>'.format(op_info.op_type, ', '.join(utensor_dtypes))
    else:
      op_type = op_info.op_type
    self.template_vars['op_type'] = op_type
    self.template_vars['op_name'] = op_name
    self.template_vars['input_map'] = input_map
    self.template_vars['output_map'] = output_map
    self.template_vars['out_shapes_map'] = out_shapes_map
    self.template_vars['out_dtypes_map'] = out_dtypes_map


class AddOpEvalSnippet(OpEvalSnippet):
  __inputs__ = ['a', 'b']
  __outputs__ = ['c']


class SimpleContainer(SnippetBase):
  __headers__ = set(['"uTensor/uTensor.hpp"', "<vector>"])
  __template_name__ = 'containers/rearch/simple.cpp'

  def __init__(self, declare_snippets=None, eval_snippests=None):
    if declare_snippets is None:
      declare_snippets = []
    if eval_snippests is None:
      eval_snippests = []
    SnippetBase.__init__(self)
    self._declare_snippets = []
    self._eval_snippests = []
    for snp in declare_snippets:
      self.add_declare_snippet(snp)
    for snp in eval_snippests:
      self.add_eval_snippet(snp)

  def add_declare_snippet(self, snippet):
    self.__headers__.update(snippet.headers)
    self._declare_snippets.append(snippet)
  
  def add_eval_snippet(self, snippet):
    self.__headers__.update(snippet.headers)
    self._eval_snippests.append(snippet)
  
  def add_header(self, header, *headers):
    self._add_header(header)
    for header in headers:
      self._add_header(header)
    return self
  
  def _add_header(self, header):
    if not header.startswith('"') and not header.startswith("<"):
      header = '"{}"'.format(header)
    self.__headers__.add(header)
  
  def render(self):
    return self.template.render(
      declare_snippets=self._declare_snippets,
      eval_snippets=self._eval_snippests,
      **self.template_vars
    )
