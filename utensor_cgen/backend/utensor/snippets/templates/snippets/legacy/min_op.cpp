{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{   
    RamTensor<{{out_dtype}}>* out_tensor;
    {% if out_shape %}
    out_tensor = new RamTensor<{{out_dtype}}>({ {%for shape in out_shape[:-1]%}{{shape}}, {%endfor%}{{out_shape[-1]}} }{%if address %}, {{address[0]}}{%endif%});
    {% else %}
    out_tensor = new RamTensor<{{out_dtype}}>({%if address %}{{address[0]}}{%endif%});
    {% endif %}
    {% if ref_count%}
    ctx.add(out_tensor, "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(out_tensor, "{{output}}");
    {% endif %}
    ctx.push(new MinOp(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
    {% if create_sptr %}
    {{sptr_name}} = ctx.get("{{output}}");
    {% endif %}
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}