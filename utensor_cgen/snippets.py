# -*- coding:utf8 -*-
from ._snippets_base import Snippet, register_template  # pylint: disable=W0611

__all__ = ["CreateTensorIdxSnippet", "CreateTensorNewSnippet", "CreateOpSnippet"]


class CreateTensorIdxSnippet(Snippet):

  def __init__(self, **kwargs):
    Snippet.__init__(self, "create_tensor_idx.cpp")


class CreateTensorNewSnippet(Snippet):

  def __init__(self, **kwargs):
    Snippet.__init__(self, "create_tensor_new.cpp")


class CreateOpSnippet(Snippet):
  def __init__(self, **kwargs):
    Snippet.__init__(self, "create_op.cpp")
