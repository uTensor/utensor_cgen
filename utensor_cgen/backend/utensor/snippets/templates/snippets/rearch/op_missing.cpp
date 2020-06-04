/*
    FIXME: {{op_type}} currently not supported, you have to fill up this section or it won't compile

    1. Declare the operators of type {{op_type}}
    2. Declare input tensors:
    {%for tensor, var_name in zip(input_tensors, input_var_names)%}
        - {{tensor.name}}, of type {{tensor.dtype}} and is named as {{var_name}} in this file
    {%endfor%}
    3. Declare output Tensors:
    {%for tensor, var_name in zip(output_tensors, out_var_names)%}
        - {{tensor.name}} is of type {{tensor.dtype}}, shape {{tensor.shape}} and should be named as {{var_name}}
        {%if quant_params_map[tensor.name]%}
            quantization parameters:
            - zero point: {{quant_params_map[tensor.name]["zero_point"]["value"]}}, {{quant_params_map[tensor.name]["zero_point"]["type_str"]}}
            - scale: {{quant_params_map[tensor.name]["scale"]["value"]}}, {{quant_params_map[tensor.name]["scale"]["type_str"]}}
            - is per tensor quantization: {{quant_params_map[tensor.name]["is_per_tensor"]}}
        {%endif%}
    {%endfor%}
    4. invoke `eval()`

    NOTE: normally, operator is implemented as template with type parameters as the data types of input/output tensors
    */
    // Don't forget to comment this out when you fill up the missing op and tensors
    static_assert(false, "{{op_type}} is not currently supported. Read information above please.");