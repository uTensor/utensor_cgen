# -*- coding:utf8 -*-
import re

from ._base import Snippet, SnippetContainerBase, SnippetBase

_STD_PATTERN = re.compile(r'^<[\w]+(.h|.hpp)?>$')


class Composer(object):

  def __init__(self, snippets=None):
    if snippets is None:
      snippets = []
    self._snippets = []
    for snp in snippets:
      self.add_snippet(snp)
    self._cached = False
    self._text = ""

  def compose(self):
    if not self._cached:
      self._text = ""
      self._compose_header()
      for snippet in self._snippets:
        self._text += snippet.render()
      self._cached = True
    return self._text

  def add_snippet(self, snippet):
    if not isinstance(snippet, (SnippetBase, Snippet, SnippetContainerBase)):
      msg = "expecting Snippet/SnippetContainerBase object, get {}".format(type(snippet))
      raise ValueError(msg)
    self._cached = False
    self._snippets.append(snippet)

  def _compose_header(self):
    unique_headers = set([])
    for snp in self._snippets:
      unique_headers.update(snp.headers)
    headers = [(0, header) if _STD_PATTERN.match(header) else (1, header) for header in unique_headers]
    headers = [t[1] for t in sorted(headers, reverse=True)]
    for header in headers:
      self._text += "#include {}\n".format(header)
    self._text += "\n\n"
