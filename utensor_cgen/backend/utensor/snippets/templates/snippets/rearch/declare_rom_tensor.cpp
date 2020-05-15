{%if static %}static {%endif%}Tensor {{tensor_var}} = new RomTensor({ {%for s in shape%}{{s}}{%if not loop.last%}, {%endif%}{%endfor%} }, {{utensor_dtype}}, {{buffer_var}});
{%if quant_params %}
    {% if quant_params['is_per_tensor']%}
    {{quant_params['zero_point']['type_str']}} {{tensor_var}}_zp = {{quant_params['zero_point']['value'][0]}};
    {{quant_params['scale']['type_str']}} {{tensor_var}}_scale = {{quant_params['scale']['value'][0]}};
    PerTensorQuantizationParams {{tensor_var}}_quant_params({{tensor_var}}_zp, {{tensor_var}}_scale);
    {%else%}
    {{quant_params['zero_point']['type_str']}} arr_{{tensor_var}}_zp[{{quant_params['zero_point']['value'].size}}] = { {%for v in quant_params['zero_point']['value']%}{{v}}{%if not loop.last%}, {%endif%}{%endfor%} };
    {{quant_params['scale']['type_str']}} arr_{{tensor_var}}_scale[{{quant_params['scale']['value'].size}}] = { {%for v in quant_params['scale']['value']%}{{v}}{%if not loop.last%}, {%endif%}{%endfor%} };
    PerChannelQuantizationParams {{tensor_var}}_quant_params(arr_{{tensor_var}}_zp, arr_{{tensor_var}}_scale);
    {%endif%}
    {{tensor_var}}->set_quantization_params({{tensor_var}}_quant_params);
{%endif%}