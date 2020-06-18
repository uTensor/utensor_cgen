#-*- coding:utf8 -*-
import importlib
import re
import sys
from pathlib import Path

import click

from utensor_cgen import __version__


def _load_plugin(path):
  path = Path(path)
  if not path.exists():
    raise RuntimeError('{} does not exists'.format(path))
  sys.path.insert(0, str(path.parent))
  mod_name = re.sub(r'.py$', '', path.name.split()[0])
  importlib.import_module(mod_name)
  sys.path.pop(0)

@click.group(name='utensor-cli')
@click.help_option('-h', '--help')
@click.version_option(
  __version__,
  '-V', '--version'
)
@click.option(
  "-p",
  "--plugin",
  multiple=True,
  help="path of the python module which will be loaded as plugin (multiple)",
  metavar="MODULE",
)
def cli(plugin):
  for module_path in plugin:
    _load_plugin(module_path)
