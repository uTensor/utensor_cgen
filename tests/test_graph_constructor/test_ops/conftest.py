from pytest import fixture


@fixture(name='ugraph')
def _ugraph():
    from utensor_cgen.ir import uTensorGraph
    return uTensorGraph(output_nodes=[])
