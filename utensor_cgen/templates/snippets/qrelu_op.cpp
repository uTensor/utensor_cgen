{
    ctx.add(new RamTensor<{{qout_dtype}}>(), "{{outputs[0]}}");
    ctx.add(new RamTensor<{{out_dtypes[0]}}>({1}), "{{outputs[1]}}");
    ctx.add(new RamTensor<{{out_dtypes[1]}}>({1}), "{{outputs[2]}}");
    ctx.push(new ReluOp<{{in_dtype}}, {{out_dtypes[0]}}, {{qout_dtype}}>(), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {% endfor %}"{{inputs[-1]}}" },
             { {% for tname in outputs[:-1]%}"{{tname}}", {% endfor %}"{{outputs[-1]}}" });
}