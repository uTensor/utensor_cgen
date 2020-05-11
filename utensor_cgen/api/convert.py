import os

from toml import loads

from utensor_cgen.backend.api import BackendManager


def convert_graph(model_file, output_nodes=None, config='utensor_cli.toml', target='utensor', model_name=None):
  from utensor_cgen.frontend import FrontendSelector

  if os.path.exists(config):
    with open(config) as fid:
      config = loads(fid.read())
  else:
    config = {}
  ugraph = FrontendSelector.parse(
    model_file, output_nodes,
    config=config,
    model_name=model_name
  )
  backend = BackendManager.get_backend(target)(config)
  backend.apply(ugraph)
  return ugraph
