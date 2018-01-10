{
    {% if ref_count %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}");
    {% endif %}
    ctx.push(new DequantizeOp(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
}