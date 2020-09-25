#!/usr/bin/env python
# -*- coding:utf8 -*-
import os

from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install

root_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root_dir, "LICENSE")) as rf:
    license = rf.read()


class _CompileFlatbuffMixin(object):
    def run(self):
        super(_CompileFlatbuffMixin, self).run()
        self._build_flatbuffer()

    def _build_flatbuffer(self):
        install_dir = self.install_platlib
        if install_dir is None:
            install_dir = os.path.abspath("utensor")


class _Install(_CompileFlatbuffMixin, _install):
    pass


class _Develop(_CompileFlatbuffMixin, _develop):
    pass


setup(
    name="utensor_cgen",
    version='1.0.0',
    cmdclass={"install": _Install, "develop": _Develop},
    description="C code generation program for uTensor",
    long_description="please go to [doc](https://utensor-cgen.readthedocs.io/en/latest/) page for more information",
    url="https://github.com/uTensor/utensor_cgen",
    author="Dboy Liao",
    author_email="qmalliao@gmail.com",
    license=license,
    packages=find_packages(),
    package_data={"utensor_cgen.backend.utensor.snippets": ["templates/*/*/*"]},
    entry_points={"console_scripts": ["utensor-cli=utensor_cgen.cli:cli"]},
    install_requires=[
        "Jinja2",
        "tensorflow==2.1.2",
        "onnx",
        "idx2numpy",
        "attrs",
        "click",
        "torch",
        "torchvision",
        "graphviz",
        "matplotlib",
        "toml",
        "flatbuffers",
        "ortools==7.6.7691",
    ],
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: MacOS X",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities",
    ],
)
