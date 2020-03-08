#-*- coding:utf8 -*-
import importlib
import os
import re
import sys
from pathlib import Path

import click
from toml import dumps, loads

from utensor_cgen import __version__
from utensor_cgen.backend.api import BackendManager
from utensor_cgen.utils import NArgsParam


def _get_pb_model_name(path):
  return os.path.basename(os.path.splitext(path)[0])

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

@cli.command(name='list-backends', help='list all available backends')
@click.help_option('-h', '--help')
def list_backends():
  click.secho('Available backends:', fg='green', bold=True)
  for backend in BackendManager.backends:
    click.secho(
      '  - {}'.format(backend), fg='green'
    )

@cli.command(name='generate-config', help='generate config toml file')
@click.help_option('-h', '--help')
@click.option('--target', required=True, help='target framework/platform')
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
    fid.write(
      '# we use string \'None\' to represent python None value\n'
      '# you should convert the string to None if you try to write extension for utensor_cgen\n'
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
  help="list of output nodes"
)
@click.option('--config', default='utensor_cli.toml', show_default=True, metavar='CONFIG.toml')
@click.option('--target',
              default='utensor',
              show_default=True,
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

def _show_ugraph(ugraph, oneline=False, ignore_unknown_op=False):
  import textwrap
  from utensor_cgen.backend.utensor.code_generator.legacy._operators import OperatorFactory

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
  click.secho(
    'topological ordered ops: {}'.format(ugraph.topo_order),
    fg='white', bold=True,
  )
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
