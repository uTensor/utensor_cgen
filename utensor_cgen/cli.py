#-*- coding:utf8 -*-
import argparse
import os

import pkg_resources

import click
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
              default='dropout,quantize,refcnt,inline,cmsisnn',
              help='optimization methods',
              metavar='METHOD,METHOD,...',
              show_default=True)
@click.option("-m", "--model-dir",
              metavar="DIR",
              default="models",
              help="ouptut directory for tensor data idx files",
              show_default=True)
def convet_graph(pb_file, output, data_dir, embed_data_dir,
                 debug_comment, output_nodes, transform_methods, model_dir):
  from utensor_cgen.code_generator import CodeGenerator

  if pb_file is None:
    raise ValueError("No pb file given")

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  # MODEL should default to pb_file
  if data_dir is None:
    data_dir = os.path.join("constants", _get_pb_model_name(pb_file))

  if output is None:
    output = _get_pb_model_name(pb_file) + ".cpp"
  model_path = os.path.join(model_dir, output)

  if embed_data_dir is None:
    embed_data_dir = os.path.join("/fs", data_dir)
  # TODO: pass transformation kwargs to codegenerator (better argument parser)
  generator = CodeGenerator(pb_file, data_dir, embed_data_dir, transform_methods, output_nodes, debug_comment)
  generator.generate(model_path)


@cli.command(name='show', help='show node names in the pb file')
@click.help_option('-h', '--help')
@click.argument('pb_file', required=True, metavar='MODEL.pb')
def show_pb_file(pb_file):
  import tensorflow as tf
  graph_def = tf.GraphDef()
  with open(pb_file, 'rb') as fid:
    graph_def.ParseFromString(fid.read())
  for node in graph_def.node:
    print(node.name)

if __name__ == '__main__':
  cli()
