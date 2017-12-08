{
    ctx.add(new RamTensor<{{qout_dtype}}>(), "{{outputs[0]}}");
    ctx.add(new RamTensor<{{range_dtype}}>({1}), "{{outputs[1]}}");
    ctx.add(new RamTensor<{{range_dtype}}>({1}), "{{outputs[2]}}");
    ctx.push(new RequantizeOp(),
             { {% for tname in inputs[:-1]%}"{{tname}}", {% endfor %}"{{inputs[-1]}}" },
             { {% for tname in outputs[:-1]%}"{{tname}}", {% endfor %}"{{outputs[-1]}}" });
}