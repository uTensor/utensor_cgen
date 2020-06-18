def test_parser(tflm_mnist_path):
    from utensor_cgen.frontend.tflite import TFLiteParser

    parser = TFLiteParser({})
    ugraph = parser.parse(tflm_mnist_path)

    assert ugraph.output_nodes, \
        'output_nodes is empty: {}'.format(ugraph.output_nodes)
    assert ugraph.topo_order, \
        'topo_order is empty: {}'.format(ugraph.topo_order)
    assert ugraph.lib_name == 'tflite'
