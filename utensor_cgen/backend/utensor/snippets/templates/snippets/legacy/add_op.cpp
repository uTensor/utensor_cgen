{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{
    {% if ref_count %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}");
    {% endif %}
    ctx.push(new AddOp<{{in_dtype}}, {{out_dtype}}>(),
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" }, 
             { "{{output}}" });
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}