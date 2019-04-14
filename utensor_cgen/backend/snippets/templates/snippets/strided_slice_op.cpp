{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{
    {% if ref_count %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}", {{ref_count}});
    {% else %}
    ctx.add(new RamTensor<{{out_dtype}}>(), "{{output}}");
    {% endif %}
    ctx.push(new StridedSliceOp<{{dtype}}>({{begin_mask}}, {{ellipsis_mask}}, {{end_mask}}, {{new_axis_mask}}, {{shrink_axis_mask}}),
             { {% for tname in inputs[:-1]%}"{{tname}}", {%endfor%}"{{inputs[-1]}}" },
             { "{{output}}" });
    {% if create_sptr %}
    {{sptr_name}} = ctx.get("{{output}}");
    {% endif %}
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}