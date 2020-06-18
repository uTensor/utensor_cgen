{{op_var_name}}
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
