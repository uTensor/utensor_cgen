# -*- coding:utf8 -*-
import os

from jinja2 import Environment, FileSystemLoader

_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates/')
_loader = FileSystemLoader(searchpath=_dir)

env = Environment(loader=_loader, trim_blocks=True, lstrip_blocks=True)
env.globals.update(zip=zip, len=len)

def model_name_filter(graph_name):
    return "".join(
        map(lambda s: s.capitalize(), graph_name.split("_"))
    )
env.filters['model_name_filter'] = model_name_filter

del _loader, _dir

# useful references
# - https://gist.github.com/wrunk/1317933/d204be62e6001ea21e99ca0a90594200ade2511e
