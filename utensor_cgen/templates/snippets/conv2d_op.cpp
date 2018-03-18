{
    {% if ref_counts %}
    ctx.add(new RamTensor<{{out_dtypes[0]}}>(), "{{outputs[0]}}", {{ref_counts[0]}});
    ctx.add(new RamTensor<{{out_dtypes[1]}}>({1}), "{{outputs[1]}}", {{ref_counts[1]}});
    ctx.add(new RamTensor<{{out_dtypes[2]}}>({1}), "{{outputs[2]}}", {{ref_counts[2]}});
    {% else %}
    ctx.add(new RamTensor<{{out_dtypes[0]}}>(), "{{outputs[0]}}");
    ctx.add(new RamTensor<{{out_dtypes[1]}}>({1}), "{{outputs[1]}}");
    ctx.add(new RamTensor<{{out_dtypes[2]}}>({1}), "{{outputs[2]}}");
    {% endif %}
    ctx.push(new QntConvOp<{{in_dtype}}, {{filter_dtype}}, {{out_dtypes[0]}}>({ {% for s in strides[:-1]%}{{s}}, {%endfor%}{{strides[-1]}} }, {{padding}}), 
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { {% for tname in outputs[:-1]%}"{{tname}}", {%endfor%}"{{outputs[-1]}}" });
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}
