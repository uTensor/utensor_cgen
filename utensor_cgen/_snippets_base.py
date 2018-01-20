# -*- coding:utf8 -*-
from abc import ABC
from copy import deepcopy

from .snippets_cfg import SNIPPET_CONFIG
from .template_env import env as _env

__all__ = ["Snippet", "SnippetContainer"]


class SnippetBase(ABC):
  __template_name__ = None

  def __init__(self):
    headers = SNIPPET_CONFIG.get(self.__template_name__, None)
    if headers is None:
      raise ValueError('unknown template name: {}'.format(self.__template_name__))
    self._headers = headers
    self.template_vars = {}


  @property
  def template_name(self):
    return self.__template_name__

  @property
  def template(self):
    if self.__template_name__ is None:
      raise ValueError('No template name: please override class attribute __template_name__')
    return _env.get_template(self.__template_name__)

  @property
  def headers(self):
    return deepcopy(self._headers)

  def add_header(self, header):
    self._headers.add(header)

  def remove_header(self, header):
    self._headers.remove(header)


class Snippet(SnippetBase):  # pylint: W0223

  def render(self):
    return self.template.render(**self.template_vars)


class SnippetContainer(SnippetBase):

  def __init__(self, snippets=None):
    SnippetBase.__init__(self)

    if snippets is None:
      snippets = []
    self._snippets = snippets
    for snp in self._snippets:
      self._headers.update(snp.headers)

  def add_snippet(self, snippet):
    """Add snippet into containers
    """
    if not isinstance(snippet, Snippet):
      msg = "expecting Snippet object, get {}".format(type(snippet))
      raise TypeError(msg)
    self._headers.update(snippet.headers)
    self._snippets.append(snippet)

  def render(self):
    return self.template.render(snippets=self._snippets, **self.template_vars)
