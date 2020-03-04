import os

import pytest


def test_legacy_utensor(mlp_ugraph):
    from utensor_cgen.backend.utensor import uTensorBackend

    this_dir = os.path.dirname(__file__)

    uTensorBackend(config={
        'utensor': {
            'backend': {
                'legacy-api': True,
                'code_generator': {
                    'model_dir': os.path.join(this_dir, 'models'),
                    'params_dir': os.path.join(this_dir, 'data'),
                },
                'graph_lower': {}
            },
        }
    }).apply(mlp_ugraph)


@pytest.mark.slow_test
def test_offlinememory(mlp_ugraph):
    from utensor_cgen.backend.utensor import uTensorOfflineMemoryPlanner

    uTensorOfflineMemoryPlanner(config={
                'size_float': 4,
                'size_int': 4,
                'size_uint8_t': 1
        }
    ).apply(mlp_ugraph)
