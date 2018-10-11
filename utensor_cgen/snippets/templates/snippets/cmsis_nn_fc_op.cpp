{
    {%if ref_counts%}
    ctx.add(new RamTensor<{{out_dtypes[0]}}>(), "{{outputs[0]}}", {{ref_counts[0]}});
    ctx.add(new RamTensor<{{out_dtypes[0]}}>({1}), "{{outputs[1]}}", {{ref_counts[1]}});
    ctx.add(new RamTensor<{{out_dtypes[0]}}>({1}), "{{outputs[2]}}", {{ref_counts[2]}});
    {%else%}
    ctx.add(new RamTensor<{{out_dtypes[0]}}>(), "{{outputs[0]}}");
    ctx.add(new RamTensor<{{out_dtypes[0]}}>({1}), "{{outputs[1]}}");
    ctx.add(new RamTensor<{{out_dtypes[0]}}>({1}), "{{outputs[2]}}");
    {%endif%}
    
    ctx.push(new FullyConnectedLayerCmsisOp<{%for dtype in in_dtypes[:-1]%}{{dtype[-1]}}, {%endfor%}{{in_dtypes[-1][-1]}}>(),
              { {%for tname in inputs[:-1] %}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
              { {%for tname in outputs[:-1] %}"{{tname}}", {%endfor%}"{{outputs[-1]}}" });
    {%if to_eval%}
    ctx.eval();
    {%endif%}
}
