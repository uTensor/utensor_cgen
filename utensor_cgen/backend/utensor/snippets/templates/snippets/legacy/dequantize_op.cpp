{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{
    {% if ref_count %}
    ctx.add(new RamTensor<{{out_dtype}}>({%if address %}{{address[0]}}{%endif%}), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<{{out_dtype}}>({%if address %}{{address[0]}}{%endif%}), "{{output}}");
    {% endif %}
    ctx.push(new DequantizeOp(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
    {% if create_sptr %}
    {{sptr_name}} = ctx.get("{{output}}");
    {% endif %}
    {%if to_eval%}
    ctx.eval();
    {%endif%}
}