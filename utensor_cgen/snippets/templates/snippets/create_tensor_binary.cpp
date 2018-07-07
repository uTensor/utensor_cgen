{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{    
    {%if ref_count%}
    ctx.add(new {{tensor_type}}<{{dtype}}>({% if tensor_shape %}{{tensor_shape}}{%endif%}, {{inline_name}}), 
    "{{tensor_name}}", 
    {{ref_count}});
    {% else %}
    ctx.add(new {{tensor_type}}<{{dtype}}>({% if tensor_shape %}{{tensor_shape}}{%endif%}, {{inline_name}}), 
    "{{tensor_name}}");
    {%endif%}
    {% if create_sptr %}
    {{sptr_name}} = ctx.get("{{tensor_name}}");
    {% endif %}
    {%if to_eval%}
    ctx.eval();
    {%endif%}
}
