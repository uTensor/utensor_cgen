from textwrap import wrap

from utensor_cgen.backend.base import Backend
from utensor_cgen.backend import BackendManager
from utensor_cgen.utils import class_property

@BackendManager.register
class DummyBackend(Backend):
    TARGET = 'dummy-backend'

    def __init__(self, config):
        if not config:
            config = self.default_config
        self.output_file = config[self.TARGET][self.COMPONENT]['output-file']

    def apply(self, ugraph):
        with open(self.output_file, 'w') as fid:
            fid.write('#include <stdio.h>\n\n')
            fid.write('int main(int argc, char* argv[]) {\n')
            fid.write('    printf("graph name: {}\\n");\n'.format(ugraph.name))
            fid.write('    printf("ops in topological sorted order:\\n");\n')
            for op_name in ugraph.topo_order:
                fid.write('    printf("    {}\\n");\n'.format(op_name))
            fid.write('    return 0;\n}')

    @class_property
    def default_config(cls):
        return {
            cls.TARGET: {
                    cls.COMPONENT: {
                    'output-file': 'list_op.c'
                }
            }
        }
