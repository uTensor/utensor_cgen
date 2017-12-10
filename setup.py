#!/usr/bin/env python3
# -*- coding:utf8 -*-
import os
from setuptools import setup, find_packages
from utensor_cgen import __version__

root_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root_dir, "README.md")) as rf:
  long_desc = rf.read()
with open(os.path.join(root_dir, "LICENSE")) as rf:
  license = rf.read()
with open(os.path.join(root_dir, "requirements.txt")) as rf:
  dependencies = [line.strip() for line in rf.readlines()]

version = __version__

setup(
  name='utensor_cgen',
  version=version,
  description="C code generation program for uTensor",
  long_description=long_desc,
  url="https://github.com/dboyliao/utensor_cgen",
  author="Dboy Liao",
  author_email="qmalliao@gmail.com",
  license=license,
  packages=find_packages(),
  include_package_data=True,
  package_data={"utensor_cgen": ["templates/*"]},
  entry_points={
      "console_scripts": [
          "utensor-cli=utensor_cgen.__main__:main"
      ]
  }
)
