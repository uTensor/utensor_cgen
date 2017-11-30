# -*- coding:utf8 -*-
import argparse
from .core import code_gen


def main():
  print("Hello utensor_cgen!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  args = vars(parser.parse_args())
  main(**args)
