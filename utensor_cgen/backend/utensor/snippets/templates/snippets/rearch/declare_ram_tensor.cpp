{%if shape%}
Tensor {{tensor_var}} = new RamTensor({ {%for s in shape%}{{s}}{%if not loop.last%}, {%endif%}{%endfor%} }, {{utensor_dtype}});
{%else%}
Tensor {{tensor_var}} = new RamTensor({{utensor_dtype}});
{%endif%}
{%if quantize_params%}
{{quantize_params["zero_point"]["type_str"]}} {{tensor_var_name}}_zp;
{{quantize_params["scale"]["type_str"]}} {{tensor_var_name}}_scale;
{%endif%}