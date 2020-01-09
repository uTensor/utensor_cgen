import os
from pytest import fixture

@fixture(scope='session', name='mlp_ugraph')
def mlp_ugraph():
    from utensor_cgen.frontend import FrontendSelector
    model_file = os.path.join(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), '..')
            ),
            'deep_mlp/simple_mnist.pb'
    )
    return FrontendSelector.parse(model_file, output_nodes=['y_pred'])
