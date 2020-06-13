from typing import Hashable

from utensor_cgen.backend.utensor import uTensorRearchGraphLower
from utensor_cgen.backend.utensor.code_generator.rearch._code_generator import \
    OperatorFactory
from utensor_cgen.backend.utensor.code_generator.rearch._operators._base import \
    _Operator
from utensor_cgen.backend.utensor.snippets.rearch import *
from utensor_cgen.backend.utensor.snippets.rearch import OpEvalSnippet
from utensor_cgen.legalizer.tflite import TFLiteLegalizer
from utensor_cgen.utils import must_return_type

# legalize all `Mean` to `MeanOperator`
TFLiteLegalizer.register_op_rename(old_name="Mean", new_name="MeanOperator")


# We will lowering all `MeanOperator` to `ReferenceOperators::MeanOperator`
@uTensorRearchGraphLower.CodgenAttributes.register("MeanOperator")
def handler(op_info):
    op_info.code_gen_attributes["namespaces"] = ("ReferenceOperators",)


class ReductionMeanEvalSnippet(OpEvalSnippet):
    __inputs__ = ["in"]
    __outputs__ = ["out"]


@OperatorFactory.register
class _ReductionMeanOperator(_Operator):
    namespaces = ("ReferenceOperators",)
    op_type = "MeanOperator"

    # the value returned by this method will be used as
    # the constrcutor parameters as is.
    # In utensor backend, it should return a tuple of string.
    # Since there is no parameters for `MeanOperator`, an empty tuple is returned
    @classmethod
    @must_return_type(Hashable)
    def get_constructor_parameters(cls, op_info):
        return tuple()

    def get_declare_snippet(self, op_var_name, tensor_var_map):
        return DeclareOpSnippet(
            op=self,
            templ_dtypes=[self.in_dtypes[0]],
            op_var_name=op_var_name,
            nested_namespaces=type(self).namespaces,
        )

    def get_eval_snippet(self, op_var_name, op_info, tensor_var_map):
        return ReductionMeanEvalSnippet(
            op_info=op_info,
            templ_dtypes=[self.in_dtypes[0]],
            op_name=op_var_name,
            tensor_var_map=tensor_var_map,
            nested_namespaces=type(self).namespaces,
        )
