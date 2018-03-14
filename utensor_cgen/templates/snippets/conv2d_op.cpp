{
    {% if ref_count %}
    ctx.add(new RamTensor<float>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<float>(), "{{output}}");
    {% endif %}
    ctx.push(new ConvOp<{{input_dtype}}, {{filter_dtype}}, {{output_dtype}}>(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}