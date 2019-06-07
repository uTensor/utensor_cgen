from utensor_cgen.transformer.pipeline import TransformerPipeline


def test_vgg1(vgg_ugraph):
    trans = TransformerPipeline([
        ('linear_reorder_v2', {}),
        ('quantize', {}),
        ('conv_pool_v2', {})
    ])
    new_ugraph = trans.transform(vgg_ugraph)
    num_conv = len(vgg_ugraph.get_ops_by_type('Conv2D'))
    num_convpool = len(new_ugraph.get_ops_by_type('QuantizedFusedConv2DMaxpool'))
    assert not new_ugraph.get_ops_by_type('Conv2D')
    assert num_conv == num_convpool

def test_vgg2(vgg_ugraph):
    trans = TransformerPipeline([
        ('linear_reorder_v2', {}),
        ('quantize', {}),
        ('conv_pool_v2', {})
    ])
    new_ugraph = trans.transform(vgg_ugraph)
    num_conv = len(vgg_ugraph.get_ops_by_type('Conv2D'))
    num_convpool = len(new_ugraph.get_ops_by_type('QuantizedFusedConv2DMaxpool'))
    assert not new_ugraph.get_ops_by_type('Conv2D')
    assert num_conv == num_convpool

def test_vgg3(vgg_ugraph):
    trans = TransformerPipeline([
        ('linear_reorder_v2', {}),
        ('quantize', {}),
        ('conv_pool_v2', {})
    ])
    new_ugraph = trans.transform(vgg_ugraph)
    num_conv = len(vgg_ugraph.get_ops_by_type('Conv2D'))
    num_convpool = len(new_ugraph.get_ops_by_type('QuantizedFusedConv2DMaxpool'))
    assert not new_ugraph.get_ops_by_type('Conv2D')
    assert num_conv == num_convpool
