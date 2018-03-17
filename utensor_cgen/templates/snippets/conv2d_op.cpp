{
    {% if ref_counts %}
    {%for tname, ref_count in zip(outputs, ref_counts)%}
    ctx.add(new RamTensor<float>(), "{{tname}}", {{ref_count}});
    {%endfor%}
    {% else %}
    {%for tname in outputs%}
    ctx.add(new RamTensor<float>(), "{{tname}}");
    {%endfor%}
    {% endif %}
    ctx.push(new QntConvOp<{{in_dtype}}, {{filter_dtype}}, {{out_dtype}}>({ {% for s in strides[:-1]%}{{s}}, {%endfor%}{{strides[-1]}} }, {{padding}}), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { {% for tname in outputs[:-1]%}"{{tname}}", {%endfor%}"{{outputs[-1]}}" });
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}