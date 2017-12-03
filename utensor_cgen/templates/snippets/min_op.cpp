{
    ctx.add(new RamTensor<{{out_dtype}}>("{{output}}"));
    ctx.push(new MinOp(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
}