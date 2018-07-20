#-*- coding:utf8 -*-
import argparse
import os

import click
import pkg_resources
from click.types import FuncParamType


def _nargs(sep=','):
  def parser(argstr):
    return argstr.split(sep)
  return parser

@click.group(name='utensor-cli')
@click.help_option('-h', '--help')
@click.version_option((pkg_resources
                       .get_distribution('utensor_cgen')
                       .version),
                       '-V', '--version')
def cli():
  pass
  

@cli.command(name='convert')
@click.help_option('-h', '--help')
@click.argument('pb_file', required=True, metavar='MODEL.pb')
@click.option('-o', '--output',
              metavar="FILE.cpp",
              help="output source file name, header file will be named accordingly. (defaults to protobuf name, e.g.: my_model.cpp)")
@click.option('-d', '--data-dir',
            default='models',
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
              type=FuncParamType(_nargs),
              metavar="NODE_NAME,NODE_NAME,...",
              required=True,
              help="list of output nodes")
@click.option("-O", "--transform-methods", 
              type=FuncParamType(_nargs),
              default='dropout,quantize,refcnt',
              help='optimization methods',
              metavar='METHOD,METHOD,...',
              show_default=True)
@click.option("-m", "--model-dir",
              metavar="DIR",
              default="models",
              help="ouptut directory for tensor data idx files",
              show_default=True)
def convet_graph(pb_file, src_fname, idx_dir, embed_data_dir,
                 debug_cmt, output_nodes, trans_methods, model_dir):
  pass


@cli.command(name='show')
@click.help_option('-h', '--help')
@click.argument('pb_file', required=True, metavar='MODEL.pb')
def show_pb_file(pb_file):
  pass


if __name__ == '__main__':
  cli()
