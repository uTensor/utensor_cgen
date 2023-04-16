import click

from utensor_cgen.utils import NArgsParam

from .main import cli


@cli.command(name='convert', help='convert graph to cpp/hpp files')
@click.help_option('-h', '--help')
@click.argument(
  'model_file',
  required=True,
  metavar='MODEL_FILE.{pb,onnx,pkl}',
)
@click.option(
  '--output-nodes',
  type=NArgsParam(),
  metavar="NODE_NAME,NODE_NAME,...",
  help="list of output nodes"
)
@click.option("--inputs-file", metavar="FILE", help="inputs file for the model (required for PyTorch)")
@click.option('--config', default='utensor_cli.toml', show_default=True, metavar='CONFIG.toml')
@click.option('--target',
              default='utensor',
              show_default=True,
              help='target framework/platform'
)
def convert_graph(model_file, output_nodes, config, target, **kwargs):
  from utensor_cgen.api.convert import convert_graph as _convert_graph

  _convert_graph(
    model_file=model_file,
    output_nodes=output_nodes,
    config=config,
    target=target,
    **kwargs
  )
  return 0
