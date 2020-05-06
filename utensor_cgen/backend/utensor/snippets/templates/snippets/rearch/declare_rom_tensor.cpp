{%if static %}static {%endif%}Tensor {{tensor_var}} = new RomTensor({ {%for s in shape%}{{s}}{%if not loop.last%}, {%endif%}{%endfor%} }, {{utensor_dtype}}, {{buffer_var}});
{%if quantize_params%}
{{quantize_params["zero_point"]["type_str"]}} {{tensor_var_name}}_zp;
{{quantize_params["scale"]["type_str"]}} {{tensor_var_name}}_scale;
{%endif%}