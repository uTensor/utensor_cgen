/*
    FIXME: {{op_type}} currently not supported, you have to fill up this section or it won't compile

    Input Tensors:
    {%for name in input_var_names%}
        {{name}}
    {%endfor%}

    Output Tensors:
    {%for name, var_name in zip(out_tensor_names, out_var_names)%}
        {{name}}: should be named as {{var_name}}
        {%if quant_params_map[name]%}

        quantization parameters
        zero point: {{quant_params_map[name]["zero_point"]["value"]}}, {{quant_params_map[name]["zero_point"]["type_str"]}}
        scale: {{quant_params_map[name]["scale"]["value"]}}, {{quant_params_map[name]["scale"]["type_str"]}}
        is per tensor quantization: quant_params_map[name]["is_per_tensor"]
        {%endif%}
    {%endfor%}
    */