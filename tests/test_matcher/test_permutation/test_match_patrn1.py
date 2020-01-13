from utensor_cgen.matcher import uTensorGraphMatcher


def test_id_match(patrn_ugraph):
    matcher = uTensorGraphMatcher(patrn_ugraph)
    matches = matcher.match_all(patrn_ugraph)
    assert len(matches) == 2, 'expecting 2 matches, get {} matches'.format(len(matches))
    assert all([not match.is_dangling for match in matches])
    match = matches[0]
    assert match.patrn2subj_op_map['input0'].name in ['input0', 'input1']
    assert match.patrn2subj_op_map['input1'].name in ['input0', 'input1']
    assert match.patrn2subj_op_map['input0'].name != match.patrn2subj_op_map['input1'].name
    assert match.patrn2subj_op_map['add0'].name == 'add0'
    assert match.patrn2subj_op_map['output'].name == 'output'
    
    assert match.subj2patrn_op_map['input0'].name in ['input0', 'input1']
    assert match.subj2patrn_op_map['input1'].name in ['input0', 'input1']
    assert match.subj2patrn_op_map['input0'].name != match.subj2patrn_op_map['input1'].name
    assert match.subj2patrn_op_map['add0'].name == 'add0'
    assert match.subj2patrn_op_map['output'].name == 'output'

    for tensor in patrn_ugraph.input_tensors:
        assert tensor.name in match.patrn2subj_tensor_map, \
            '{} is missing'.format(tensor.name)
    for tensor in patrn_ugraph.output_tensors:
        assert tensor.name in match.subj2patrn_tensor_map, \
            '{} is missing'.format(tensor.name)

def test_match_sub1(patrn_ugraph, subject_ugraph1):
    matcher = uTensorGraphMatcher(patrn_ugraph)
    matches = matcher.match_all(subject_ugraph1)
    assert all([not match.is_dangling for match in matches])
    assert matches, 'expecting matches, get {} matches'.format(len(matches))
    match = matches[0]
    assert len(matches) == 2, 'should be exactly two match, get {}'.format(len(matches))
    assert match.patrn2subj_op_map['input0'].name in ['sub_input0', 'sub_input1'], match
    assert match.patrn2subj_op_map['input1'].name in ['sub_input0', 'sub_input1'], match
    assert match.patrn2subj_op_map['input0'].name != match.patrn2subj_op_map['input1'].name
    assert match.patrn2subj_op_map['add0'].name == 'sub_add0', match
    assert match.patrn2subj_op_map['output'].name == 'sub_add1', match

def test_match_sub1_1(patrn_ugraph, subject_ugraph1_1):
    matcher = uTensorGraphMatcher(patrn_ugraph)
    matches = matcher.match(subject_ugraph1_1)
    assert matches, 'expecting matches, get {} matches'.format(len(matches))
    assert all([not match.is_dangling for match in matches])
    match = matches[0]
    assert match.patrn2subj_op_map['input0'].name in ['sub_input0', 'sub_input1']
    assert match.patrn2subj_op_map['input1'].name in ['sub_input0', 'sub_input1']
    assert match.patrn2subj_op_map['add0'].name == 'sub_add0'
    assert match.patrn2subj_op_map['output'].name == 'sub_add1'

def test_match_sub1_2(patrn_ugraph, subject_ugraph1_2):
    matcher = uTensorGraphMatcher(patrn_ugraph)
    matches = matcher.match(subject_ugraph1_2)
    assert all([not match.is_dangling for match in matches])
    assert matches, 'expecting matches, get {} matches'.format(len(matches))
    match = matches[0]
    assert match.patrn2subj_op_map['input0'].name in ['sub_input0', 'sub_input1']
    assert match.patrn2subj_op_map['input1'].name in ['sub_input0', 'sub_input1']
    assert match.patrn2subj_op_map['add0'].name == 'sub_add0'
    assert match.patrn2subj_op_map['output'].name == 'sub_add1'
