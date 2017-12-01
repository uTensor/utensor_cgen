{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{
    TensorIdxImporter t_import;
    {% if tensor_shape %}
    {{ctx_var_name}}.add(new {{tensor_type}}<{{dtype}}>({{tensor_shape}}, "{{tensor_name}}"), {{init_count}});
    {% else %}
    {{ctx_var_name}}.add(new {{tensor_type}}<{{dtype}}>("{{tensor_name}}"), {{init_count}});
    {% endif %}

    {% if create_sptr %}
    {{sptr_name}} = {{ctx_var_name}}.get("{{tensor_name}}");
    {% endif %}
}