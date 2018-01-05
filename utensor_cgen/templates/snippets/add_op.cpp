{
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}", {{init_count}});
    ctx.push(new AddOp<{{in_dtype}}, {{out_dtype}}>(),
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" }, 
             { "{{output}}" });
}