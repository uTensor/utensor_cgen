# -*- coding:utf8 -*-
from copy import deepcopy


class Snippet(object):

  def __init__(self, headers=None):
    if headers is None:
      headers = []
    self._headers = headers
  
  @property
  def headers(self):
    return deepcopy(self._headers)

  def __str__(self):
    return ""

  def __radd__(self, other):
    if isinstance(other, str):
      return other + str(self)
    return NotImplemented

class SnippetContainer(Snippet):

  def __init__(self, *args, **kwargs):
    super(SnippetContainer, self).__init__(*args, **kwargs)
    self._snippets = []
    self._text = ""
    self._cached = False

  def add_snippet(self, snippet):
    """Add snippet into containers
    """
    self._cached = False
    if not isinstance(snippet, Snippet):
      msg = "expecting Snippet object, get {}".format(type(snippet))
      raise TypeError(msg)
    self._snippets.append(snippet)
  
  def reset(self):
    self._text = ""
    self._cached = False
    self._snippets = []

  def __str__(self):
    raise NotImplementedError("Not implemented")

class MainContainer(SnippetContainer):

  def __str__(self):
    if not self._cached:
      self._text = "int main(int argc, char* argv[]) {\n"
      for snippet in self._snippets:
        self._text += str(snippet)
      self._text += "    return 0;\n}"
    return self._text


class HelloWorld(Snippet):

  def __init__(self):
    self._headers = ["<stdio.h>"]

  def __str__(self):
    return '    printf("Hello world!\\n");\n'


class TensorSnippet(Snippet):

  def __init__(self):
    pass
