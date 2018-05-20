# -*- coding:utf8 -*-
# pylint: disable=C0301
from __future__ import print_function

import argparse
import os
import pkg_resources
import sys


def _get_pb_model_name(path):
  return os.path.basename(os.path.splitext(path)[0])


def main(pb_file, src_fname, idx_dir, embed_data_dir,
         debug_cmt, output_nodes, method, model_dir):
  if pb_file is None:
    raise ValueError("No pb file given")

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  # MODEL should default to pb_file
  if idx_dir is None:
    idx_dir = os.path.join("constants", _get_pb_model_name(pb_file))

  if src_fname is None:
    src_fname = _get_pb_model_name(pb_file) + ".cpp"
  model_path = os.path.join(model_dir, src_fname)

  from .code_generator import CodeGenerator

  if embed_data_dir is None:
    embed_data_dir = os.path.join("/fs", idx_dir)
  generator = CodeGenerator(pb_file, idx_dir, embed_data_dir, method, output_nodes, debug_cmt)
  generator.generate(model_path)


def _nargs(sep=','):
  def parser(argstr):
    return argstr.split(sep)
  return parser


def _build_parser():
  pkg_version = pkg_resources.get_distribution('utensor_cgen').version
  parser = argparse.ArgumentParser()
  parser.add_argument("pb_file", metavar='MODEL.pb',
                      help="input protobuf file")
  parser.add_argument("-d", "--data-dir", dest='idx_dir',
                      metavar="DIR",
                      help="ouptut directory for tensor data idx files (defaults to protobuf name, e.g.: constants/my_model)")
  parser.add_argument("-m", "--model-dir", dest='model_dir',
                      metavar="DIR", default="models",
                      help="ouptut directory for tensor data idx files (default: %(default)s)")
  parser.add_argument("-o", "--output", dest="src_fname",
                      metavar="FILE.cpp",
                      help="output source file name, header file will be named accordingly. (defaults to protobuf name, e.g.: my_model.cpp)")
  parser.add_argument("-D", "--embed-data-dir", dest="embed_data_dir",
                      metavar="EMBED_DIR", default=None,
                      help="the data dir on the develop board (default: the value as the value of -d/data-dir flag)")
  parser.add_argument("--output-nodes", dest="output_nodes",
                      type=_nargs(), metavar="NODE_NAME,NODE_NAME,...",
                      required=True,
                      help="list of output nodes (required)")
  parser.add_argument("-O", "--optimize-method", choices=['None', 'refcnt'],
                      dest='method', default='refcnt',
                      help='optimization method (default: %(default)s)')
  parser.add_argument("--debug-comment", dest="debug_cmt",
                      action="store_true",
                      help="Add debug comments in the output source file (default: %(default)s)")
  parser.add_argument("-v", "--version", action="version",
                      version="utensor-cli {}".format(pkg_version),
                      help="show version")
  return parser


def cli():
  parser = _build_parser()
  args = vars(parser.parse_args())
  main(**args)


if __name__ == "__main__":
    cli()
