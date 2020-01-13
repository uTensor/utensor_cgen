from utensor_cgen.transformer.conv_pool import ConvPoolTransformer


def factory():
    def test(vgg_ugraph_pair):
        vgg_ugraph, num_layers = vgg_ugraph_pair
        trans = ConvPoolTransformer()
        new_ugraph = trans.transform(vgg_ugraph)
        num_convpool = len(new_ugraph.get_ops_by_type('QuantizedFusedConv2DMaxpool'))
        assert not new_ugraph.get_ops_by_type('Conv2D')
        assert num_layers == num_convpool, num_layers
    return test

# 5 random tests
test_vgg1 = factory()
test_vgg2 = factory()
test_vgg3 = factory()
test_vgg4 = factory()
test_vgg5 = factory()
