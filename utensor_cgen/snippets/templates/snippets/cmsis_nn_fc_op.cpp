{
    {% if ref_counts%}
    ctx.add(new RamTensor<{{dtypes[-1]}}>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<{{dtypes[-1]}}>(), "{{output}}");
    {% endif %}
    ctx.push(new FullyConnectedLayerCmsisOp<{%for dtype in dtypes[:-1]%}{{dtype}}, {%endfor%}{{dtypes[-1]}}>(),
              { {%for tname in inputs[:-1] %}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
              { "{{output}}" });
    {%if to_eval%}
    ctx.eval();
    {%endif%}
}
