#!/usr/bin/env python
# -*- coding:utf8 -*-
import os

from setuptools import find_packages, setup

root_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root_dir, "LICENSE")) as rf:
    license = rf.read()

setup(
    name='utensor_cgen',
    version_config={
        "starting_version": "0.0.0",
        "version_format": "{tag}.dev{sha:.7s}"
    },
    setup_requires=['better-setuptools-git-version'],
    description="C code generation program for uTensor",
    long_description="please go to [doc](https://utensor-cgen.readthedocs.io/en/latest/) page for more information",
    url="https://github.com/dboyliao/utensor_cgen",
    author="Dboy Liao",
    author_email="qmalliao@gmail.com",
    license=license,
    packages=find_packages(),
    include_package_data=True,
    package_data={"utensor_cgen": ["backend/snippets/templates/*"]},
    entry_points={
        "console_scripts": [
            "utensor-cli=utensor_cgen.cli:cli"
        ]},
    install_requires=[
        'Jinja2',
        'tensorflow==1.13.1',
        'idx2numpy',
        'attrs',
        'click',
        'torch',
        'torchvision',
        'onnx-tf==1.2.1',
        'graphviz'
    ],
    extras_require={
        'dev': ['pytest']
    },
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
        "Topic :: Utilities"
    ]
)
