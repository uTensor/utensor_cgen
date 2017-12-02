# -*- coding:utf8 -*-
# pylint: disable=C0301
import argparse
import os
from .core import CodeGenerator


def _main(pb_file, src_fname, idx_dir, mbed_data_dir):
  if mbed_data_dir is None:
    mbed_data_dir = os.path.join("/fs", idx_dir)
  generator = CodeGenerator(pb_file, idx_dir)
  generator.generate(src_fname, mbed_data_dir)


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
  parser.add_argument("-D", "--embed-data-dir", dest="mbed_data_dir",
                      metavar="EMBED_DIR", default=None,
                      help="the data dir on the develop board (default: the value as the value of -d/data-dir flag)")
  return parser


def main():
  parser = _build_parser()
  args = vars(parser.parse_args())
  _main(**args)


if __name__ == "__main__":
    main()
