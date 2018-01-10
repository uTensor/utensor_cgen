{    
    ctx.add(new {{tensor_type}}<{{dtype}}>({% if tensor_shape %}{{tensor_shape}}{%endif%}), "{{tensor_name}}", {%if ref_count%}{{ref_count}}{%endif%});
}