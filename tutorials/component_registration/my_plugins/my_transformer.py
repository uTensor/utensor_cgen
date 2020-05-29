from copy import deepcopy

from utensor_cgen.transformer import (GENERIC_SENTINEL, Transformer,
                                      TransformerPipeline)


@TransformerPipeline.register_transformer
class MyAddTransformer(Transformer):
    KWARGS_NAMESCOPE = 'myadd_transformer'
    METHOD_NAME = 'myadd_transformer'
    # this transformer is generic and can be applied to any graph
    APPLICABLE_LIBS = GENERIC_SENTINEL
    
    def transform(self, ugraph):
        new_ugraph = deepcopy(ugraph)
        for op_info in new_ugraph.get_ops_by_type('AddOperator'):
            op_info.op_type = 'MyAddOperator'
        return new_ugraph
