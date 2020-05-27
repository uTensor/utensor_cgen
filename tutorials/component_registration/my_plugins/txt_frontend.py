import os

import numpy as np

from utensor_cgen.frontend import FrontendSelector, Parser
from utensor_cgen.ir import OperationInfo, TensorInfo, uTensorGraph
from utensor_cgen.ir.converter import (AttrValueConverter,
                                       GenericTensorConverterMixin)
from utensor_cgen.utils import topologic_order_graph


@FrontendSelector.register(['.txt'])
class TxtParser(Parser):
    def parse(self, txt_file, output_nodes=None):
        graph_name, _ = os.path.splitext(
            os.path.basename(txt_file)
        )
        if output_nodes is None:
            output_nodes = []
        add_all_nodes = not output_nodes
        ugraph = uTensorGraph(name=graph_name, output_nodes=output_nodes, lib_name='txtlib')
        with open(txt_file, 'r') as fid:
            for line in fid:
                try:
                    op_name, value = line.split(' ', maxsplit=1)
                except Exception:
                    raise ValueError('invalid line: {}'.format(line))
                value = np.array(eval(value))
                out_tensor = TensorInfo(
                    '{}:0'.format(op_name),
                    op_name,
                    dtype=value.dtype,
                    shape=list(value.shape),
                    ugraph=ugraph
                )
                op_info = OperationInfo(
                    name=op_name,
                    lib_name='txtlib',
                    ugraph=ugraph,
                    input_tensors=[],
                    output_tensors=[out_tensor],
                    op_type='Const',
                    op_attr={
                        "value": AttrValueConverter.GenericType(
                            value_name="tensor",
                            value=GenericTensorConverterMixin.GenericType(
                                np_array=value
                            ),
                        )
                    }
                )
                if add_all_nodes:
                    ugraph.output_nodes.append(op_name)
        topologic_order_graph(ugraph)
        return ugraph
