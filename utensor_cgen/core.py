# -*- coding:utf8 -*-
from .pbparser import parse_pb

__all__ = ["code_gen"]


def code_gen(pb_file: str) -> (str, str):
  graph_info, layers = parse_pb(pb_file)
  output_src = ""
  output_header = ""
  return output_src, output_header
