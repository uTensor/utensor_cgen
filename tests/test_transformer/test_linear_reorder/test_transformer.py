def test_linear_reorder_1(subj_ugraph_simple):
    from utensor_cgen.transformer.linear_reoder import LinearReorderTransformerV2

    transformer = LinearReorderTransformerV2()
    new_ugraph = transformer.transform(subj_ugraph_simple)
    for op in new_ugraph.get_ops_by_type('Relu'):
        assert op.input_tensors[0].op.op_type == 'MaxPool'

def test_linear_reorder_2(subj_ugraph_no_effect):
    from utensor_cgen.transformer.linear_reoder import LinearReorderTransformerV2

    transformer = LinearReorderTransformerV2()
    # this transform should have no effect on the graph
    new_ugraph = transformer.transform(subj_ugraph_no_effect)
    for op in new_ugraph['output'].input_nodes:
        assert op.op_type == 'MaxPool'
        assert op.op_attr['ksize'].value.ints_value == [1, 3, 3, 1]
        assert op.op_attr['strides'].value.ints_value == [1, 2, 2, 1]
        relu_op = op.input_nodes[0]
        assert relu_op.op_type == 'Relu'
        assert relu_op.output_tensors[0].shape == [None, 512, 512, 10]
