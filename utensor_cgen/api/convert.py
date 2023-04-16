import os

from toml import loads

from utensor_cgen.backend.api import BackendManager
from utensor_cgen.logger import logger


def convert_graph(model_file,
  output_nodes=None,
  config='utensor_cli.toml',
  target='utensor',
  model_name=None,
  **kwargs
):
  from utensor_cgen.frontend import FrontendSelector

  if os.path.exists(config):
    logger.info('config file {} found, reading configurations'.format(config))
    with open(config) as fid:
      config = loads(fid.read())
  else:
    config = {}
  ugraph = FrontendSelector.parse(
    model_file,
    config,
    output_nodes,
    model_name=model_name,
    **kwargs
  )
  backend = BackendManager.get_backend(target)(config)
  backend.apply(ugraph)
  return ugraph
