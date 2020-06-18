from utensor_cgen.frontend.onnx import OnnxParser


def test_onnx_parser(onnx_model_path):
    parser = OnnxParser({})
    ugraph = parser.parse(onnx_model_path)

    assert ugraph.lib_name == 'onnx'
    assert ugraph.output_nodes
    assert ugraph.topo_order
    assert ugraph.ops_info
