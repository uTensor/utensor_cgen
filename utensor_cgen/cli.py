#-*- coding:utf8 -*-
import os
import sys
from pathlib import Path

import click
from click.types import Choice
from toml import dumps, loads

from utensor_cgen import __version__
from utensor_cgen.backend.api import BackendManager
from utensor_cgen.utils import NArgsParam


def _get_pb_model_name(path):
  return os.path.basename(os.path.splitext(path)[0])

def _load_transformer(path):
  # FIXME: better way to activate user plugins other than exec
  from utensor_cgen.transformer import TransformerPipeline, Transformer

  _globals = {}
  transform_plugin = Path(path).absolute()
  with transform_plugin.open('r') as fid:
    exec(fid.read(), _globals)
  for obj in _globals.values():
    if obj is Transformer:
      continue
    if isinstance(obj, type) and issubclass(obj, Transformer):
      TransformerPipeline.register_transformer(obj)


@click.group(name='utensor-cli')
@click.help_option('-h', '--help')
@click.version_option(__version__,
                      '-V', '--version')
@click.option("--transform-plugin",
              default=None,
              help="path of the python file which user-defined transformers live",
              metavar="MODULE.py",
)
def cli(transform_plugin):
  if transform_plugin is not None:
    _load_transformer(transform_plugin)

@cli.command(name='generate-config', help='generate config toml file')
@click.help_option('-h', '--help')
@click.option('--target', type=Choice(BackendManager.backends), required=True, help='target framework/platform')
@click.option('-o', '--output', default='utensor_cli.toml', metavar='CONFIG.toml', help='the output config file name')
def generate_config(target, output):
  backend_cls = BackendManager.get_backend(target)
  config = backend_cls.default_config
  click.secho(
    'generating config file: {}'.format(output),
    fg='white',
    bold=True,
  )
  with open(output, 'w') as fid:
    fid.write(
      '# https://github.com/toml-lang/toml\n'
      '# <target_name>.<component>.<part>\n'
    )
    fid.write(dumps(config))

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
  required=True,
  help="list of output nodes"
)
@click.option('--config', default='utenosr_cli.toml', show_default=True, metavar='CONFIG.toml')
@click.option('--target',
              default='utensor',
              show_default=True, 
              type=Choice(BackendManager.backends),
              help='target framework/platform'
)
def convert_graph(model_file, output_nodes, config, target):
  from utensor_cgen.frontend import FrontendSelector

  if os.path.exists(config):
    with open(config) as fid:
      config = loads(fid.read())
  else:
    config = {}
  ugraph = FrontendSelector.parse(model_file, output_nodes, config)
  backend = BackendManager.get_backend(target)(config)
  backend.apply(ugraph)

@cli.command(name='list-trans-methods', help='list all available graph transformation')
@click.help_option('-h', '--help')
@click.option('--verbose', is_flag=True)
def list_trans_methods(verbose):
  from utensor_cgen.transformer import TransformerPipeline

  if verbose:
    for name, trans_cls in TransformerPipeline.TRANSFORMER_MAP.items():
      click.secho(name, fg='white', bold=True)
      click.secho(trans_cls.__doc__, fg='yellow', bold=True)
  else:
    click.secho(
      str(TransformerPipeline.all_transform_methods()),
      fg='white', bold=True
    )
  return 0

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
@click.argument('model_file', required=True, metavar='MODEL.{pb,pkl}')
def show_graph(model_file, **kwargs):
  _, ext = os.path.splitext(model_file)
  output_nodes = kwargs.pop('output_nodes')
  if ext == '.pb' or ext == '.pbtxt':
    _show_pb_file(model_file, output_nodes=output_nodes, **kwargs)
  elif ext == '.pkl':
    import pickle
    with open(model_file, 'rb') as fid:
      ugraph = pickle.load(fid)
    _show_ugraph(ugraph, **kwargs)
  else:
    msg = click.style('unknown file extension: {}'.format(ext), fg='red', bold=True)
    click.echo(msg, file=sys.stderr)

def _show_pb_file(pb_file, output_nodes, **kwargs):
  import tensorflow as tf
  from utensor_cgen.frontend.tensorflow import GraphDefParser
  ugraph = GraphDefParser.parse(pb_file, output_nodes=output_nodes)
  _show_ugraph(ugraph, **kwargs)

def _show_ugraph(ugraph, oneline=False, ignore_unknown_op=False):
  import textwrap
  from utensor_cgen.backend.utensor._operators import OperatorFactory

  unknown_ops = set([])
  if oneline:
    tmpl = click.style("{op_name} ", fg='yellow', bold=True) + \
      "op_type: {op_type}, inputs: {inputs}, outputs: {outputs}"
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      msg = tmpl.format(op_name=op_name, op_type=op_info.op_type,
                        inputs=[tensor.name for tensor in op_info.input_tensors],
                        outputs=[tensor.name for tensor in op_info.output_tensors])
      click.echo(msg)
      if not OperatorFactory.is_supported(op_info.op_type):
        unknown_ops.add(op_info)
  else:
    tmpl = click.style('op_name: {op_name}\n', fg='yellow', bold=True) + \
    '''\
      op_type: {op_type}
      input(s):
        {inputs}
        {input_shapes}
      ouptut(s):
        {outputs}
        {output_shapes}
    '''
    tmpl = textwrap.dedent(tmpl)
    paragraphs = []
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      op_str = tmpl.format(
        op_name=op_name,
        op_type=op_info.op_type,
        inputs=op_info.input_tensors,
        outputs=op_info.output_tensors,
        input_shapes=[tensor.shape for tensor in op_info.input_tensors],
        output_shapes=[tensor.shape for tensor in op_info.output_tensors])
      paragraphs.append(op_str)
      if not OperatorFactory.is_supported(op_info.op_type):
        unknown_ops.add(op_info)
    click.echo('\n'.join(paragraphs))
  if unknown_ops and not ignore_unknown_op:
    click.echo(
      click.style('Unknown Ops Detected', fg='red', bold=True)
    )
    for op_info in unknown_ops:
      click.echo(
        click.style('    {}: {}'.format(op_info.name, op_info.op_type), fg='red')
      )
  return 0

if __name__ == '__main__':
  cli()
