{%if quantize_params_map%}
{%for tensor_var in output_map.values()%}
{%if tensor_var in quantize_params_map%}
{{tensor_var}}_zp = {{quantize_params_map[tensor_var]["zero_point"]["value"]}};
{{tensor_var}}_scale = {{quantize_params_map[tensor_var]["scale"]["value"]}};
{%endif%}
{%endfor%}
{%endif%}

    {
        dws_params.padding = {{padding}};
        dws_params.stride_width = {{stride_width}};
        dws_params.stride_height = {{stride_height}};
        dws_params.depth_multiplier = {{depth_multiplier}};
        dws_params.activation = {{activation}};
        dws_params.dilation_width_factor = {{dilation_width_factor}};
        dws_params.dilation_height_factor = {{dilation_height_factor}};
        {{op_var_name}}.set_params(dws_params);
    }
    {{op_var_name}}
    ].set_inputs({
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
