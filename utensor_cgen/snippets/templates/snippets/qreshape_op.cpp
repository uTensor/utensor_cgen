{
    {% if ref_counts%}
    ctx.add(new RamTensor<uint8_t>(), "{{outputs[0]}}", {{ref_counts[0]}});
    ctx.add(new RamTensor<float>({1}), "{{outputs[1]}}", {{ref_counts[1]}});
    ctx.add(new RamTensor<float>({1}), "{{outputs[2]}}", {{ref_counts[2]}});
    {% else %}
    ctx.add(new RamTensor<uint8_t>(), "{{outputs[0]}}");
    ctx.add(new RamTensor<float>({1}), "{{outputs[1]}}");
    ctx.add(new RamTensor<float>({1}), "{{outputs[2]}}");
    {% endif %}
    ctx.push(new QuantizedReshapeOp(),
              { {%for tname in inputs[:-1] %}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
              { {%for tname in outputs[:-1] %}"{{tname}}", {%endfor%}"{{outputs[-1]}}" });
    {%if to_eval%}
    ctx.eval();
    {%endif%}
}