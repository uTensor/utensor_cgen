# -*- coding:utf8 -*-
from ._snippets_base import Snippet


class CreateTensorSnippet(Snippet):

  def __init__(self, template_name, template_vars=None):
    Snippet.__init__(self, template_name, template_vars)
