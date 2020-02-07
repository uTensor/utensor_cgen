{%for tensor_var, shape in out_shapes_map.items()%}
Tensor {{tensor_var}} = new RamTensor({ {%for s in shape[:-1]%}{{s}}, {%endfor%}{{shape[-1]}} }, {{out_dtypes_map[tensor_var]}});
{%endfor%}
    {{op_name}}
        .set_inputs({
{%for name, tensor_var in input_map.items()%}
            { {{op_type}}::{{name}}, {{tensor_var}} },
{%endfor%}
        })
        .set_outputs({
{%for name, tensor_var in output_map.items()%}
            { {{op_type}}::{{name}}, {{tensor_var}}}
{%endfor%}
        })
        .eval();
