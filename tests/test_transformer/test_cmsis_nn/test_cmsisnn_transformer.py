from utensor_cgen.transformer import CMSIS_NN_Transformer
from utensor_cgen.ir import uTensorGraph

def test_cmsisnn_trnasformer(fusion_graph_tuple):
    (ugraph, ugraph1) = fusion_graph_tuple
    transformer = CMSIS_NN_Transformer()
    test_graph = transformer.transform(ugraph1)