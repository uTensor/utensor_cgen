def test_linear_reorder_1(subj_ugraph_1):
    from utensor_cgen.transformer.linear_reoder import LinearReorderTransformerV2

    transformer = LinearReorderTransformerV2()
    assert not transformer.prune_graph
    new_ugraph = transformer.transform(subj_ugraph_1)
    for op in new_ugraph['output'].input_nodes:
        assert op.op_type == 'Relu', 'expecting Relu, get {}'.format(op.op_type)
        pool_op = op.input_nodes[0]
        assert pool_op.op_type == 'MaxPool', \
            'expecting MaxPool as input of Relu, get {}'.format(pool_op.op_type)
        assert pool_op.op_attr['ksize'].value.ints_value == [1, 3, 3, 1]
        assert pool_op.op_attr['strides'].value.ints_value == [1, 2, 2, 1]
        assert pool_op.input_tensors[0].shape == [None, 512, 512, 10]
