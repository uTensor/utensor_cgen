{
    {%if ref_counts%}
    ctx.add(new RamTensor<{{out_dtype}}>({1}), "{{outputs[0]}}", {{ref_counts[0]}});
    ctx.add(new RamTensor<{{out_dtype}}>({1}), "{{outputs[1]}}", {{ref_counts[1]}});
    {%else%}
    ctx.add(new RamTensor<{{out_dtype}}>({1}), "{{outputs[0]}}");
    ctx.add(new RamTensor<{{out_dtype}}>({1}), "{{outputs[1]}}");
    {%endif%}
    
    ctx.push(new QuantRangeForMultiplicationOp<uint8_t, uint8_t, {{out_dtype}}>(),
              { {%for tname in inputs[:-1] %}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
              { {%for tname in outputs[:-1] %}"{{tname}}", {%endfor%}"{{outputs[-1]}}" });
    {%if to_eval%}
    ctx.eval();
    {%endif%}
}
