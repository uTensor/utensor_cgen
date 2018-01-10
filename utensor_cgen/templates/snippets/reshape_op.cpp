{
    ctx.add(new RamTensor<float>(), "{{output}}", {{ref_count}});
    ctx.push(new ReshapeOp(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
}