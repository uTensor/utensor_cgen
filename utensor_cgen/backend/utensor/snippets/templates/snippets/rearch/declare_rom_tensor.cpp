{%if static %}static {%endif%}Tensor {{tensor_var}} = new RomTensor({ {%for s in shape%}{{s}}{%if not loop.last%}, {%endif%}{%endfor%} }, {{utensor_dtype}}, {{buffer_var}});
{%if quant_params %}
    {{quant_params['zero_point']['type_str']}} {{tensor_var}}_zp = {{quant_params['zero_point']['value']}};
    {{quant_params['scale']['type_str']}} {{tensor_var}}_scale = {{quant_params['scale']['value']}};
    PerTensorQuantizationParams {{tensor_var}}_quant_params({{tensor_var}}_zp, {{tensor_var}}_scale);
    {{tensor_var}}->set_quantization_params({{tensor_var}}_quant_params);
{%endif%}