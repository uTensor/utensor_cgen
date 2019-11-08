{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{
    {% if ref_count %}
    ctx.add(new RamTensor<q7_t>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<q7_t>(), "{{output}}");
    {% endif %}
    ctx.push(new Uint8Q7OriginOp(),
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" }, 
             { "{{output}}" });
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}