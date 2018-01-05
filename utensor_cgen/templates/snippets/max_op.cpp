{   {%if out_shape %}
    ctx.add(new RamTensor<{{out_dtype}}>({ {%for shape in out_shape[:-1]%}{{shape}}, {%endfor%}{{out_shape[-1]}} }), "{{output}}", {{init_count}});
    {%else%}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}", {{init_count}});
    {%endif%}
    ctx.push(new MaxOp(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
}