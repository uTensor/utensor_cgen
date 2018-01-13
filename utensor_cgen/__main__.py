# -*- coding:utf8 -*-
# pylint: disable=C0301
import argparse
import os

def main(pb_file, src_fname, idx_dir, embed_data_dir, debug_cmt, output_nodes):
  from .core import CodeGenerator

  if embed_data_dir is None:
    embed_data_dir = os.path.join("/fs", idx_dir)
  generator = CodeGenerator(pb_file, idx_dir, embed_data_dir, debug_cmt, output_nodes)
  generator.generate(src_fname)


def _nargs(sep=','):
  def parser(argstr):
    print(argstr)
    return argstr.split(sep)
  return parser


def _build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("pb_file", metavar='MODEL.pb',
                      help="input protobuf file")
  parser.add_argument("-d", "--data-dir", dest='idx_dir',
                      metavar="DIR", default="idx_data",
                      help="ouptut directory for tensor data idx files (default: %(default)s)")
  parser.add_argument("-o", "--output", dest="src_fname",
                      metavar="FILE.cpp", default="model.cpp",
                      help="output source file name, header file will be named accordingly. (default: %(default)s)")
  parser.add_argument("-D", "--embed-data-dir", dest="embed_data_dir",
                      metavar="EMBED_DIR", default=None,
                      help="the data dir on the develop board (default: the value as the value of -d/data-dir flag)")
  parser.add_argument("--output-nodes", dest="output_nodes",
                      type=_nargs(), metavar="node,node,...",
                      default=None,
                      help="list of output nodes")
  parser.add_argument("--debug-comment", dest="debug_cmt",
                      action="store_true",
                      help="Add debug comments in the output source file (default: %(default)s)")
  return parser


def cli():
  parser = _build_parser()
  args = vars(parser.parse_args())
  main(**args)


if __name__ == "__main__":
    cli()
