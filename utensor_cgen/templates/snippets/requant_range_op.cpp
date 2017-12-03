{
    ctx.add(new RamTensor<{{out_dtype}}>("{{outputs[0]}}"));
    ctx.add(new RamTensor<{{out_dtype}}>("{{outputs[1]}}"));
    ctx.push(new Requantization_RangeOp(),
             { {%for tname in inputs[:-1]%}{{tname}}, {% endfor %}{{inputs[-1]}} },
             { {%for tname in outputs[:-1]%}{{tname}}, {% endfor %}{{outputs[-1]}} });
}