{
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{outputs[0]}}");
    ctx.add(new RamTensor<float>({1}), "{{outputs[1]}}");
    ctx.add(new RamTensor<float>({1}), "{{outputs[2]}}");
    ctx.push(new QntMatMulOp<{{x_dtype}}, {{w_dtype}}, {{out_dtype}}>(), 
             { {%for tname in inputs[:-1] %}"{{tname}}", {% endfor %} "{{inputs[-1]}}" },
             { {%for tname in outputs[:-1] %}"{{tname}}", {% endfor %} "{{outputs[-1]}}" });
}