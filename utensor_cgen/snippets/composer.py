# -*- coding:utf8 -*-
import re

from ._snippets import Snippet, SnippetContainerBase

_STD_PATTERN = re.compile(r'^<[\w]+(.h|.hpp)?>$')


class Composer(object):

  def __init__(self, snippets=None):
    if snippets is None:
      snippets = []
    for snp in snippets:
      if not isinstance(snp, (Snippet, SnippetContainerBase)):
        msg = "expecting Snippet/SnippetContainerBase objects, get {}".format(type(snp))
        raise TypeError(msg)
    self._snippets = snippets
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
    if not isinstance(snippet, (Snippet, SnippetContainerBase)):
      msg = "expecting Snippet/SnippetContainerBase object, get {}".format(type(snippet))
      raise ValueError(msg)
    self._cached = False
    self._snippets.append(snippet)

  def _compose_header(self):
    unique_headers = set([])
    for snp in self._snippets:
      unique_headers.update(snp.headers)
    headers = [(header, 0) if _STD_PATTERN.match(header) else (header, 1) for header in unique_headers]
    headers = [t[0] for t in sorted(headers, key=lambda t: t[1], reverse=True)]
    for header in headers:
      self._text += "#include {}\n".format(header)
    self._text += "\n\n"
