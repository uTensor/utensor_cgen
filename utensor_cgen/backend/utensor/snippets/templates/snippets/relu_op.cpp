{% if create_sptr %}
S_TENSOR {%for sptr_name in sptr_names[:-1]%}{{sptr_name}}, {%endfor%} {{sptr_names[-1]}};
{% endif %}
{
    {%if ref_count%}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}", {{ref_count}});
    {%else%}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}");
    {%endif%}
    ctx.push(new ReluOp<{{in_dtype}}, {{out_dtype}}>(),
             { {% for tname in inputs[:-1]%}"{{tname}}", {% endfor %}"{{inputs[-1]}}" },
             { "{{output}}" });
    {% for sptr_name, output in zip(sptr_names, outputs) %}
    {{sptr_name}} = ctx.get("{{output}}");
    {% endfor %}
    {% if to_eval%}
    ctx.eval();
    {% endif %}
}