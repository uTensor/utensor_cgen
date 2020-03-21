import os

from pytest import fixture

test_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..'
    )
)

@fixture(scope='session', name='mlp_ugraph')
def mlp_ugraph():
    from utensor_cgen.frontend import FrontendSelector
    model_file = os.path.join(
        test_dir,
        'deep_mlp/simple_mnist.pb'
    )
    return FrontendSelector.parse(model_file, output_nodes=['y_pred'])

@fixture(scope='session', name='simple_ugraph')
def simple_ugraph():
    from utensor_cgen.frontend import FrontendSelector
    model_file = os.path.join(
        test_dir,
        'simple_graph.pb'
    )
    return FrontendSelector.parse(model_file, output_nodes=['u'])
