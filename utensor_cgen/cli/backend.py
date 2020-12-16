import os
from pprint import pformat

import click

from utensor_cgen import __version__
from utensor_cgen.api.backend import generate_config as _generate_config
from utensor_cgen.api.backend import get_backends, get_trans_methods
from utensor_cgen.backend.api import BackendManager

from .main import cli


@cli.command(name='list-backends', help='list all available backends')
@click.help_option('-h', '--help')
def list_backends():
  backends = get_backends()
  click.secho('Available backends:', fg='green', bold=True)
  for backend in backends:
    click.secho(
      '  - {}'.format(backend), fg='green'
    )
  return 0

@cli.command(name='list-trans-methods', help='list all available graph transformation')
@click.help_option('-h', '--help')
@click.option('--verbose', is_flag=True)
def list_trans_methods(verbose):
  from pprint import pformat

  trans_methods = get_trans_methods()

  if verbose:
    for name, trans_cls in trans_methods.items():
      click.secho(name, fg='white', bold=True)
      click.secho(trans_cls.__doc__, fg='yellow', bold=True)
  else:
    click.secho(
      pformat(list(trans_methods.keys())),
      fg='white', bold=True
    )
  return 0

@cli.command(name='list-support-ops', help='list all supported op in the backend')
@click.help_option('-h', '--help')
@click.option('--target', default='utensor', show_default=True)
@click.option('--config', default='utensor_cli.toml', show_default=True)
def list_support_ops(target, config):
  from utensor_cgen.backend.api import BackendManager
  if os.path.exists(config):
    backend = BackendManager.get_backend(target).from_file(config)
  else:
    backend = BackendManager.get_backend(target)({})
  click.secho(
    f"Supported ops in {target!r}:",
    fg="green",
    bold=True,
  )
  click.secho(
    pformat(backend.support_ops),
    fg='white',
    bold=True
  )

@cli.command(name='generate-config', help='generate config toml file')
@click.help_option('-h', '--help')
@click.option('--target', required=True, help='target framework/platform')
@click.option('-o', '--output', default='utensor_cli.toml', metavar='CONFIG.toml', help='the output config file name')
def generate_config(target, output):
  _generate_config(target, output)
  click.secho(
    'config file generated: {}'.format(output),
    fg='white',
    bold=True,
  )
  return 0
