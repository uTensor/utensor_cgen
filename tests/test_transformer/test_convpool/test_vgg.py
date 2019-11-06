from utensor_cgen.transformer.pipeline import TransformerPipeline


def factory():
    def test(vgg_ugraph):
        trans = TransformerPipeline([
            ('linear_reorder', {}),
            ('quantize', {}),
            ('conv_pool', {})
        ])
        new_ugraph = trans.transform(vgg_ugraph)
        num_conv = len(vgg_ugraph.get_ops_by_type('Conv2D'))
        num_convpool = len(new_ugraph.get_ops_by_type('QuantizedFusedConv2DMaxpool'))
        assert not new_ugraph.get_ops_by_type('Conv2D')
        assert num_conv == num_convpool
    return test

# 5 random tests
test_vgg1 = factory()
test_vgg2 = factory()
test_vgg3 = factory()
test_vgg4 = factory()
test_vgg5 = factory()
