import os

def test_utensor(mlp_ugraph):
    from utensor_cgen.backend.utensor import uTensorBackend

    this_dir = os.path.dirname(__file__)

    uTensorBackend(config={
        'utensor': {
            'backend': {
                'code_generator': {
                    'model_dir': os.path.join(this_dir, 'models'),
                    'params_dir': os.path.join(this_dir, 'data'),
                },
                'graph_lower': {}
            },
        }
    }).apply(mlp_ugraph)


def test_offlinememory(mlp_ugraph):
    from utensor_cgen.backend.utensor import uTensorOfflineMemoryPlanner

    uTensorOfflineMemoryPlanner(config={
                'size_float': 4,
                'size_int': 4,
                'size_uint8_t': 1
        }
    ).apply(mlp_ugraph)