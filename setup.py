#!/usr/bin/env python
# -*- coding:utf8 -*-
import sysconfig
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install

root_dir = Path(__file__).parent
with (root_dir / "LICENSE").open("r") as rf:
    LICENSE = rf.read()


class _Install(_install):
    pass


class _Develop(_develop):
    pass


def _get_version():
    version_file_path = Path(__file__).parent / "utensor_cgen" / "_version.py"
    if not version_file_path.exists():
        print(f"WARN: no {version_file_path} found, using 0.0.0 as __version__")
        return "0.0.0"
    ns = {}
    with version_file_path.open("r") as fid:
        exec(fid.read(), {}, ns)
    return ns["__version__"]


def _get_install_requires():
    if "arm64" in sysconfig.get_platform():
        return [
            "tensorflow-macos",
            "onnx",
            "ortools",
            "jinja2",
            "idx2numpy",
            "attrs",
            "click",
            "torch",
            "torchvision",
            "graphviz",
            "matplotlib",
            "toml",
            "flatbuffers",
        ]
    else:
        # x86_64
        return [
            "Jinja2==3.1.2",
            "tensorflow==2.5.3",
            "protobuf<4.0.0",  # required, to make sure `ortools` dependency won't mess up with TF
            # https://github.com/onnx/onnx/issues/2808#issuecomment-771064232
            "onnx==1.8.1",
            "idx2numpy==1.2.3",
            "attrs==20.3.0",
            "click==7.1.2",
            "torch==1.7.0",
            "torchvision==0.8.1",
            "graphviz==0.15",
            "matplotlib==3.3.3",
            "toml==0.10.2",
            "ortools==8.2.8710",
        ]


setup(
    name="utensor_cgen",
    version=_get_version(),
    cmdclass={"install": _Install, "develop": _Develop},
    description="C code generation program for uTensor",
    long_description="please go to [doc](https://utensor-cgen.readthedocs.io/en/latest/) page for more information",
    url="https://github.com/uTensor/utensor_cgen",
    author="Dboy Liao",
    author_email="qmalliao@gmail.com",
    license=LICENSE,
    packages=find_packages(),
    package_data={"utensor_cgen.backend.utensor.snippets": ["templates/*/*/*"]},
    entry_points={"console_scripts": ["utensor-cli=utensor_cgen.cli:cli"]},
    install_requires=_get_install_requires(),
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
