{
    {#
    // {%if ref_counts%}
    // ctx.add(new RamTensor<{{out_dtypes[0]}}>(), "{{outputs[0]}}", {{ref_counts[0]}});
    // ctx.add(new RamTensor<{{out_dtypes[0]}}>({1}), "{{outputs[1]}}", {{ref_counts[1]}});
    // ctx.add(new RamTensor<{{out_dtypes[0]}}>({1}), "{{outputs[2]}}", {{ref_counts[2]}});
    // {%else%}
    // ctx.add(new RamTensor<{{out_dtypes[0]}}>(), "{{outputs[0]}}");
    // ctx.add(new RamTensor<{{out_dtypes[0]}}>({1}), "{{outputs[1]}}");
    // ctx.add(new RamTensor<{{out_dtypes[0]}}>({1}), "{{outputs[2]}}");
    // {%endif%}
    #}

    {%if ref_counts%}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}", {{ref_counts[0]}});
    {%else%}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}");
    {%endif%}
    
    ctx.push(new FullyConnectedLayerCmsisOp<{{out_dtype}}>(),
              { {%for tname in inputs[:-1] %}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
              { "{{output}}" });
    {%if to_eval%}
    ctx.eval();
    {%endif%}
}
