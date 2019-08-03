# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from better_setuptools_git_version import get_version

# from utensor_cgen.cli import _version as version

sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'utensor_cgen'
copyright = '2019, uTensor Team'
author = 'dboyliao, Neil Tan, kazami, Michael Bartling'
version = get_version()
master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension'
]
autodoc_typehints = 'none'
autoapi_dirs = ['../../utensor_cgen']
autoapi_generate_api_docs = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster' # 'nature'
add_function_parentheses = False

# autodoc config
autodoc_default_options = {
    'member-order': 'bysource',
    'no-undoc-members': True
}
# these will cause readthedoc build process fiail
# see https://github.com/readthedocs/readthedocs.org/issues/5328 
autodoc_mock_imports = [
    'idx2numpy',
    'tensorflow',
    'torch',
    'torchvision',
    'onnx-tf',
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
