{
    ctx.add(new RamTensor<{{out_dtype}}>({1}), "{{outputs[0]}}", {{init_counts[0]}});
    ctx.add(new RamTensor<{{out_dtype}}>({1}), "{{outputs[1]}}", {{init_counts[1]}});
    ctx.push(new Requantization_RangeOp(),
             { {%for tname in inputs[:-1]%}"{{tname}}", {% endfor %}"{{inputs[-1]}}" },
             { {%for tname in outputs[:-1]%}"{{tname}}", {% endfor %}"{{outputs[-1]}}" });
}