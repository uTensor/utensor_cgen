#!/usr/bin/env python
# -*- coding:utf8 -*-
import os

from setuptools import find_packages, setup

root_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root_dir, "README.md")) as rf:
    long_desc = rf.read()
with open(os.path.join(root_dir, "LICENSE")) as rf:
    license = rf.read()

setup(
    name='utensor_cgen',
    version_format='{tag}.dev{commitcount}+{gitsha}',
    setup_requires=['setuptools-git-version'],
    description="C code generation program for uTensor",
    long_description=long_desc,
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
        'tensorflow',
        'idx2numpy',
        'attrs',
        'click',
        'torch',
        'torchvision',
        'onnx-tf',
    ],
    extras_require={
        'dev': ['pytest', 'graphviz']
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
