{
    {% if ref_counts%}
    ctx.add(new RamTensor<uint8_t>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<uint8_t>(), "{{output}}");
    {% endif %}
    ctx.push(new FullyConnectedLayerCmsisOp(),
              { {%for tname in inputs[:-1] %}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
              { "{{output}}" });
    {%if to_eval%}
    ctx.eval();
    {%endif%}
}