{   
    RamTensor* out_tensor;
    {%if out_shape %}
    out_tensor = new RamTensor<{{out_dtype}}>({ {%for shape in out_shape[:-1]%}{{shape}}, {%endfor%}{{out_shape[-1]}} });
    {%else%}
    out_tensor = new RamTensor<{{out_dtype}}>();
    {%endif%}
    {%if ref_count%}
    ctx.add(out_tensor, "{{output}}", {{ref_count}});
    {%else%}
    ctx.add(out_tensor, "{{output}}");
    {%endif%}
    ctx.push(new MinOp(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
}