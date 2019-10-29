from pytest import fixture


@fixture(name='ugraph')
def _ugraph():
    from utensor_cgen.ir import uTensorGraph
    return uTensorGraph(output_nodes=[])

@fixture(name='quant_trans')
def _quant_trans():
    from utensor_cgen.transformer.quantize import QuantizeTransformer
    return QuantizeTransformer()
