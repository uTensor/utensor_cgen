# -*- coding:utf8 -*-
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import wraps
import re

from tensorflow import GraphDef, NodeDef

from ..parser import OperationInfo, TensorInfo
from ..parser.utils import parse_tensor_name

__all__ = ["NamescopeTransformer", "Transformer"]


class Transformer(object):
  __metaclass__ = ABCMeta

  def __new__(cls, *args, **kwargs):
    self = object.__new__(cls)
    ori_transform = self.transform

    @wraps(ori_transform)
    def transform(graph_def, output_nodes):
      trans_graph_def = ori_transform(graph_def, output_nodes)
      return self._prune_graph_def(trans_graph_def, output_nodes)

    self.transform = transform
    return self

  @abstractmethod
  def transform(self, graph_def, output_nodes):
    raise NotImplementedError('You should overwrite transform method for all transformer')

  @classmethod
  def _prune_graph_def(cls, graph_def, output_nodes):
    all_in_ops = set(output_nodes)
    for node in graph_def.node:
      in_ops = [parse_tensor_name(tname)[0] for tname in node.input]
      all_in_ops.update(in_ops)
    new_nodes = []
    for node in graph_def.node:
      if node.name not in all_in_ops:
        continue
      new_node = NodeDef()
      new_node.CopyFrom(node)
      new_nodes.append(new_node)
    new_graph_def = GraphDef()
    new_graph_def.CopyFrom(graph_def)
    while new_graph_def.node:
      new_graph_def.node.pop()
    new_graph_def.node.extend(new_nodes)
    return new_graph_def


class _DropoutTransformer(Transformer):
  """Remove Dropout Op
  """
  _NAMESCOPE_PATTERN = re.compile(r'(dropout[_\w\d]*)/.*')

  def transform(self, graph_def, output_nodes):
    new_graph_def = GraphDef()
    new_nodes = []
    dropout_input_map = self._find_input(graph_def)
    for node in graph_def.node:
      match = self._NAMESCOPE_PATTERN.match(node.name)
      if match:
        # ignore all dropout nodes
        continue
      new_node = NodeDef()
      new_node.CopyFrom(node)
      # replace inputs with dropout inputs
      tnames = [tname for tname in new_node.input]
      for i, tname in enumerate(tnames):
        op_name = parse_tensor_name(tname)[0]
        match = self._NAMESCOPE_PATTERN.match(op_name)
        if match:
          name_scope = match.group(1)
          dropout_in = dropout_input_map[name_scope]
          new_node.input.pop(i)
          new_node.input.insert(i, dropout_in)
      new_nodes.append(new_node)
    new_graph_def.node.extend(new_nodes)
    return new_graph_def

  def _find_dropout_clusters(self, graph_def):
    clusters = defaultdict(lambda: [])
    for node in graph_def.node:
      match = self._NAMESCOPE_PATTERN.match(node.name)
      if match:
        name_scope = match.group(1)
        clusters[name_scope].append(node.name)
    return dict(clusters)

  def _find_input(self, graph_def):
    clusters = self._find_dropout_clusters(graph_def)
    input_map = {}
    for node in graph_def.node:
      match = self._NAMESCOPE_PATTERN.match(node.name)
      if match:
        name_scope = match.group(1)
        cluster = clusters[name_scope]
        in_ops = [parse_tensor_name(in_tensor)[0] for in_tensor in node.input]
        for in_op in in_ops:
          if in_op not in cluster and not in_op.startswith('keep_prob'):
            input_map[name_scope] = in_op
    return input_map


class _BatchNormTransformer(Transformer):
  """Replace Batch Norm namescope with uTensor Op
  """

  _NAMESCOPE_PATTERN = re.compile(r'(BatchNorm[_\w\d]*)/.*')

  def transform(self, graph_def, output_nodes):
    # TODO implement this!
    pass


class NamescopeTransformer(Transformer):

  _DELEGATION_MAP = {
    "dropout": _DropoutTransformer,
    "BatchNorm": _BatchNormTransformer
  }

  def __init__(self, target_name_scope):
    if target_name_scope not in self._DELEGATION_MAP:
      raise ValueError('Unsupport namescope: {}'.format(target_name_scope))
    self._target_ns = target_name_scope
    self._delegate = self._DELEGATION_MAP[target_name_scope]()

  def transform(self, graph_def, output_nodes):
    new_graph_def = self._delegate.transform(graph_def, output_nodes)
    return new_graph_def

  @classmethod
  def register_transformer(cls, target_name_scope, transformer, overwrite=False):
    assert isinstance(transformer, Transformer), \
      "expecting Transformer object, get {}".format(type(transformer))
    assert target_name_scope not in cls._DELEGATION_MAP or overwrite, \
      "Registering existing transformer without overwriting"
    cls._DELEGATION_MAP[target_name_scope] = transformer

  def __repr__(self):
    return "NamescopeTransformer('{}')".format(self._target_ns)
