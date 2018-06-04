{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{    
	{% if arr %}		
	const {{dtype}} arr[ {{tensor_length}} ] = { {% for val in valarr %} {{val[0]}}, {% endfor %} }
	{% else %}
	const {{dtype}} arr [ {{tensor_length}} ] = { {% for val in valarr %} {{val}}, {% endfor %} }
	{% endif %}
    ctx.add(new {{tensor_type}}<{{dtype}}>({% if tensor_shape %}{{tensor_shape}}{%endif%}), "{{tensor_name}}", {%if ref_count%}{{ref_count}}{%endif%});
    {% if create_sptr %}
    {{sptr_name}} = ctx.get("{{tensor_name}}");
    {% endif %}
    {%if to_eval%}
    ctx.eval();
    {%endif%}
}
