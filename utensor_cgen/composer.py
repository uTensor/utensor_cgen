# -*- coding:utf8 -*-
import re
from .snippets import Snippet, SnippetContainer, MainContainer

_STD_PATTERN = re.compile(r"^<[\w\.]+h?>$")


class Composer(object):

  def __init__(self, snippets=None, container=None):
    if snippets is None:
      snippets = []
    if container is None:
      container = MainContainer()
    for snp in snippets:
      if not isinstance(snp, Snippet):
        msg = "expecting Snippet objects, get {}".format(type(snp))
        raise TypeError(msg)
    if not isinstance(container, SnippetContainer):
      msg = "expecting SnippetContainer objects, get {}".format(type(container))
      raise TypeError(msg)

    self._container = container
    self._snippets = snippets
    self._cached = False
    self._text = ""

  def compose(self):
    if not self._cached:
      self._text = ""
      self._compose_header()
      for snippet in self._snippets:
        self._container.add_snippet(snippet)
      self._text += str(self._container)
      self._cached = True
    return self._text

  def add_snippet(self, snippet):
    if not isinstance(snippet, Snippet):
      msg = "expecting Snippet object, get {}".format(type(snippet))
      raise ValueError(msg)
    self._cached = False
    self._snippets.append(snippet)

  def _compose_header(self):
    unique_headers = set([])
    for snp in self._snippets:
      unique_headers.update(snp.headers)
    unique_headers.update(self._container.headers)
    headers = [(header, 0) if _STD_PATTERN.match(header) else (header, 1) for header in unique_headers]
    headers = [t[0] for t in sorted(headers, key=lambda t: t[1], reverse=True)]
    for header in headers:
      self._text += "#include {}\n".format(header)
    self._text += "\n\n"
