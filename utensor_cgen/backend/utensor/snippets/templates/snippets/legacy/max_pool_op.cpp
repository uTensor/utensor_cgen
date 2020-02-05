{% if create_sptr %}
S_TENSOR {%for sptr_name in sptr_names[:-1]%}{{sptr_name}}, {%endfor%} {{sptr_names[-1]}};
{% endif %}
{
    {% if ref_count %}
    ctx.add(new RamTensor<{{dtype}}>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<{{dtype}}>(), "{{output}}");
    {% endif %}

    ctx.push(new MaxPoolingOp<{{dtype}}>({{wind_rows}}, {{wind_cols}}, {{row_stride}}, {{col_stride}}, {{padding}}),
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });

    {# {% if create_sptr %} #}
    {% for sptr_name, output in zip(sptr_names, outputs) %}
    {{sptr_name}} = ctx.get("{{output}}");
    {% endfor %}
    {# {% endif %} #}

    {% if to_eval %}
    ctx.eval();
    {% endif %}
}