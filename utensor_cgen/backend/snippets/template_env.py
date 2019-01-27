# -*- coding:utf8 -*-
from jinja2 import Environment, PackageLoader

_loader = PackageLoader('utensor_cgen', 'backend/snippets/templates')

env = Environment(loader=_loader, trim_blocks=True, lstrip_blocks=True)
env.globals.update(zip=zip)

del _loader

# useful references
# - https://gist.github.com/wrunk/1317933/d204be62e6001ea21e99ca0a90594200ade2511e
