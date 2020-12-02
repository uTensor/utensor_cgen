#-*- coding:utf8 -*-
import importlib
import re
import sys
from pathlib import Path

import click

from utensor_cgen import __version__
from utensor_cgen.logger import logger


def _load_plugin(path_or_name):
  path = Path(path_or_name)
  if path.exists():
    # load as plugin
    logger.info(
      f'plugin found at path {path_or_name!r}, loading plugin'
    )
    sys.path.insert(0, str(path.parent))
    mod_name = re.sub(r'.py$', '', path.name.split()[0])
    importlib.import_module(mod_name)
    sys.path.pop(0)
  else:
    # load as extension
    # an extension should follow naming convention: utensor_<extension name>
    logger.info(f"trying to load {path_or_name!r} as extension")
    try:
      ext_name = f'utensor_{path_or_name}'
      importlib.import_module(ext_name)
    except ImportError as err:
      raise RuntimeError(f'Fail to load plugin/extension: {ext_name}') from err
  logger.info(f"{path_or_name!r} loaded")

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
