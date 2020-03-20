{
    {% if ref_count %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}");
    {% endif %}
    ctx.push(new ConvOp<{{in_dtype}}, {{filter_dtype}}, {{out_dtype}}>({ {% for s in strides[:-1]%}{{s}}, {%endfor%}{{strides[-1]}} }, {{padding}}),
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}"});
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}
