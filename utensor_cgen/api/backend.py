from toml import dumps

from utensor_cgen import __version__
from utensor_cgen.backend.api import BackendManager


def get_backends():
  return BackendManager.backends

def get_trans_methods():
  from utensor_cgen.transformer import TransformerPipeline

  return TransformerPipeline.TRANSFORMER_MAP

def generate_config(target, output='utensor_cli.toml'):
  backend_cls = BackendManager.get_backend(target)
  config = backend_cls.default_config
  
  with open(output, 'w') as fid:
    fid.write(
      '# utensor-cli version {}\n'.format(__version__) + \
      '# https://github.com/toml-lang/toml\n' + \
      '# <target_name>.<component>.<part>\n'
    )
    fid.write(
      '# we use string \'None\' to represent python None value\n'
      '# you should convert the string to None if you try to write extension for utensor_cgen\n'
    )
    fid.write(dumps(config))
