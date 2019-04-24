#-*- coding:utf8 -*-
import os
import sys

import click
import pkg_resources

from .utils import NArgsParam


def _get_pb_model_name(path):
  return os.path.basename(os.path.splitext(path)[0])

@click.group(name='utensor-cli')
@click.help_option('-h', '--help')
@click.version_option((pkg_resources
                       .get_distribution('utensor_cgen')
                       .version),
                       '-V', '--version')
def cli():
  pass


@cli.command(name='convert', help='convert graph to cpp/hpp files')
@click.help_option('-h', '--help')
@click.argument('pb_file', required=True, metavar='MODEL.pb')
@click.option('-o', '--output',
              metavar="FILE.cpp",
              help="output source file name, header file will be named accordingly. (defaults to protobuf name, e.g.: my_model.cpp)")
@click.option('-d', '--data-dir',
              metavar='DIR',
              help="ouptut directory for tensor data idx files",
              show_default=True)
@click.option('-D', '--embed-data-dir',
              metavar='EMBED_DIR',
              help=("the data dir on the develop board "
                    "(default: the value as the value of -d/data-dir flag)"))
@click.option('--debug-comment',
              is_flag=True,
              help="Add debug comments in the output source file",
              show_default=True)
@click.option("--output-nodes",
              type=NArgsParam(),
              metavar="NODE_NAME,NODE_NAME,...",
              required=True,
              help="list of output nodes")
@click.option("--transform-methods",
              type=NArgsParam(),
              default='dropout,quantize,inline,biasAdd,remove_id_op,refcnt',
              help='optimization pipeline',
              metavar='METHOD,METHOD,...',
              show_default=True)
@click.option("-m", "--model-dir",
              metavar="DIR",
              default="models",
              help="ouptut directory for tensor data idx files",
              show_default=True)
@click.option("--save-graph",
              is_flag=True,
              help="save transformed graph")
def convert_graph(pb_file, output, data_dir, embed_data_dir, save_graph,
                  debug_comment, output_nodes, transform_methods, model_dir):
  from utensor_cgen.backend import CodeGenerator

  if pb_file is None:
    raise ValueError("No pb file given")

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  # MODEL should default to pb_file
  if data_dir is None:
    data_dir = os.path.join("constants", _get_pb_model_name(pb_file))

  if output is None:
    output =  "{}.cpp".format(_get_pb_model_name(pb_file))
  model_path = os.path.join(model_dir, output)

  if embed_data_dir is None:
    embed_data_dir = os.path.join("/fs", data_dir)
  # TODO: pass transformation kwargs to codegenerator (better argument parser)
  generator = CodeGenerator(pb_file, data_dir, embed_data_dir,
                            transform_methods, output_nodes,
                            save_graph, debug_comment)
  generator.generate(model_path)


@cli.command(name='show', help='show node names in the pb file')
@click.help_option('-h', '--help')
@click.option('--oneline', is_flag=True,
              help='show in oneline format (no detail information)')
@click.argument('model_file', required=True, metavar='MODEL.{pb,pkl}')
def show_graph(model_file, **kwargs):
  _, ext = os.path.splitext(model_file)
  if ext == '.pb':
    _show_pb_file(model_file, **kwargs)
  elif ext == '.pkl':
    import pickle
    with open(model_file, 'rb') as fid:
      ugraph = pickle.load(fid)
    _show_ugraph(ugraph, **kwargs)
  else:
    msg = click.style('unknown file extension: {}'.format(ext), fg='red', bold=True)
    click.echo(msg, file=sys.stderr)

def _show_pb_file(pb_file, **kwargs):
  import tensorflow as tf
  from utensor_cgen.frontend.tensorflow import GraphDefParser

  ugraph = GraphDefParser.parse(pb_file)
  _show_ugraph(ugraph, **kwargs)

def _show_ugraph(ugraph, oneline=False):
  import textwrap
  if oneline:
    tmpl = click.style("{op_name} ", fg='yellow', bold=True) + \
      "op_type: {op_type}, inputs: {inputs}, outputs: {outputs}"
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      msg = tmpl.format(op_name=op_name, op_type=op_info.op_type,
                        inputs=[tensor.name for tensor in op_info.input_tensors],
                        outputs=[tensor.name for tensor in op_info.output_tensors])
      click.echo(msg)
  else:
    tmpl = click.style('op_name: {op_name}\n', fg='yellow', bold=True) + \
    '''\
      op_type: {op_type}
      input(s):
        {inputs}
      ouptut(s):
        {outputs}
    '''
    tmpl = textwrap.dedent(tmpl)
    paragraphs = []
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      op_str = tmpl.format(
        op_name=op_name,
        op_type=op_info.op_type,
        inputs=op_info.input_tensors,
        outputs=op_info.output_tensors)
      paragraphs.append(op_str)
    click.echo('\n'.join(paragraphs))
  return 0

if __name__ == '__main__':
  cli()
