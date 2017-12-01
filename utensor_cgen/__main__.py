# -*- coding:utf8 -*-
# pylint: disable=C0301
import argparse
from .core import CodeGenerator


def main(pb_file, src_fname, idx_dir):
  generator = CodeGenerator(pb_file, idx_dir)
  generator.generate(src_fname)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='utensor_cgen')
  parser.add_argument("pb_file", metavar='MODEL.pb',
                      help="input protobuf file")
  parser.add_argument("-d", "--data-dir", dest='idx_dir',
                      metavar="DIR", default="idx_data",
                      help="ouptut directory for tensor data idx files (default: %(default)s)")
  parser.add_argument("-o", "--output", dest="src_fname",
                      metavar="FILE.cpp", default="model.cpp",
                      help="output source file name, header file will be named accordingly. (default: %(default)s)")
  args = vars(parser.parse_args())
  main(**args)
