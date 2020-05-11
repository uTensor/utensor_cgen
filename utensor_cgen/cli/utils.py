import os

import click
from toml import loads

from utensor_cgen.api.utils import show_ugraph as _show_ugraph
from utensor_cgen.utils import NArgsParam

from .main import cli


@cli.command(name='show', help='show node names in the pb file')
@click.help_option('-h', '--help')
@click.option('--oneline', is_flag=True,
              help='show in oneline format (no detail information)')
@click.option('--ignore-unknown-op', is_flag=True,
              help='ignore unknown/unsupported ops')
@click.option("--output-nodes",
              type=NArgsParam(),
              metavar="NODE_NAME,NODE_NAME,...",
              help="list of output nodes")
@click.option('--config', default='utensor_cli.toml', show_default=True, metavar='CONFIG.toml')
@click.argument('model_file', required=True, metavar='MODEL.{pb,pkl}')
def show_graph(model_file, config, **kwargs):
  import pickle
  from utensor_cgen.frontend import FrontendSelector

  _, ext = os.path.splitext(model_file)
  output_nodes = kwargs.pop('output_nodes')

  if ext == '.pkl':
    with open(model_file, 'rb') as fid:
      ugraph = pickle.load(fid)
    _show_ugraph(ugraph, **kwargs)
    return 0

  if os.path.exists(config):
    with open(config) as fid:
      config = loads(fid.read())
  else:
    config = {}
  try:
    ugraph = FrontendSelector.parse(model_file, output_nodes, config)
    _show_ugraph(ugraph, **kwargs)
    return 0
  except RuntimeError as err:
    msg = err.args[0]
    click.secho(msg, fg='red', bold=True)
    return 1
