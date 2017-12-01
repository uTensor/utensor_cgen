#!/usr/bin/env python3
# -*- coding:utf8 -*-
import argparse
import sys
from utensor_cgen._snippets_base import Snippet, SnippetContainer
from utensor_cgen.composer import Composer


def main(output_fname):
  """Main function
  """
  comp = Composer()
  hello_world = Snippet("hello_world.cpp")
  main_container = SnippetContainer("main.cpp", [hello_world, hello_world])
  comp.add_snippet(main_container)
  with open(output_fname, "w") as wf:
    wf.write(comp.compose())
  return 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-o", "--output", dest="output_fname",
                      default="main.cpp", help="output file name (default: %(default)s)", 
                      metavar="FILE.cpp")
  ARGS = vars(parser.parse_args())
  sys.exit(main(**ARGS))
