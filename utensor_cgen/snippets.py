# -*- coding:utf8 -*-
import os
from copy import deepcopy
from jinja2 import Template
from .snippets_cfg import SNIPPET_CONFIG, CONAINER_CONFIG

_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
_TEMPLATE_DIR = os.path.join(_ROOT_DIR, "templates")

_SNIPPET_DIR = os.path.join(_TEMPLATE_DIR, "snippets")
_CONTAINER_DIR = os.path.join(_TEMPLATE_DIR, "containers")

_SNIPPETS_FILES = os.listdir(_SNIPPET_DIR)
_SNIPPETS = dict((fname, os.path.join(_SNIPPET_DIR, fname)) for fname in _SNIPPETS_FILES)
_CONTAINER_FILES = os.listdir(_CONTAINER_DIR)
_CONTAINERS = dict((fname, os.path.join(_CONTAINER_DIR, fname)) for fname in _CONTAINER_FILES)


# TODO `Snippet` and `SnippetContainer` behave really alike.
# Should be able do some refactoring here....
class Snippet(object):

  def __init__(self, template_name, template_vars=None):
    if template_name not in SNIPPET_CONFIG:
      raise ValueError("unknown tempalte name: {}".format(template_name))
    template_path = _SNIPPETS[template_name]
    if template_vars is None:
      template_vars = {}
    with open(template_path) as rf:
      template = Template(rf.read())
    self._template = template
    self._headers = SNIPPET_CONFIG[template_name]
    self.template_vars = template_vars

  @property
  def template(self):
    return self._template

  @property
  def headers(self):
    return deepcopy(self._headers)

  def render(self):
    return self._template.render(self.template_vars)

  @classmethod
  def get_template_names(cls):
    return list(SNIPPET_CONFIG.keys())


class SnippetContainer(object):

  def __init__(self, template_name, snippets=None, template_vars=None):
    if template_name not in CONAINER_CONFIG:
      raise ValueError("unknown container tempalte name: {}".format(template_name))
    template_path = _CONTAINERS[template_name]
    if template_vars is None:
      template_vars = {}
    with open(template_path) as rf:
      template = Template(rf.read())
    if snippets is None:
      snippets = []

    self._snippets = snippets
    self._template = template
    self._headers = CONAINER_CONFIG[template_name]
    for snp in self._snippets:
      self._headers.update(snp.headers)
    self.template_vars = template_vars

  @property
  def template(self):
    return self._template

  @property
  def headers(self):
    return deepcopy(self._headers)

  def add_snippet(self, snippet):
    """Add snippet into containers
    """
    if not isinstance(snippet, Snippet):
      msg = "expecting Snippet object, get {}".format(type(snippet))
      raise TypeError(msg)
    self._snippets.append(snippet)

  def render(self):
    return self._template.render(snippets=self._snippets, **self.template_vars)
