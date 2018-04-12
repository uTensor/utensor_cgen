{% if create_sptr %}
S_TENSOR {%for sptr_name in sptr_names[:-1]%}{{sptr_name}}, {%endfor%} {{sptr_names[-1]}};
{% endif %}
{
    {% if ref_counts %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{outputs[0]}}", {{ref_counts[0]}});
    ctx.add(new RamTensor<float>({1}), "{{outputs[1]}}", {{ref_counts[1]}});
    ctx.add(new RamTensor<float>({1}), "{{outputs[2]}}", {{ref_counts[2]}});
    {% else %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{outputs[0]}}");
    ctx.add(new RamTensor<float>({1}), "{{outputs[1]}}");
    ctx.add(new RamTensor<float>({1}), "{{outputs[2]}}");
    {% endif %}
    ctx.push(new QuantizedAddOp<{{x_dtype}}, {{w_dtype}}, {{out_dtype}}>(), 
             { {%for tname in inputs[:-1] %}"{{tname}}", {% endfor %} "{{inputs[-1]}}" },
             { {%for tname in outputs[:-1] %}"{{tname}}", {% endfor %} "{{outputs[-1]}}" });
    {% for sptr_name, output in zip(sptr_names, outputs) %}
    {{sptr_name}} = ctx.get("{{output}}");
    {% endfor %}
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}
