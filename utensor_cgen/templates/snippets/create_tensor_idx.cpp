{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{
    TensorIdxImporter t_import;
    ctx.add(t_import.{{importer_dtype}}_import("{{data_dir}}/{{idx_fname}}"), "{{tensor_name}}", {{init_count}});
    {% if create_sptr %}
    {{sptr_name}} = ctx.get("{{tensor_name}}");
    {% endif %}
}