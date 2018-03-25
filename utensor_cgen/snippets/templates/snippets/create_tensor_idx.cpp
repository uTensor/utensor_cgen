{% if create_sptr %}
S_TENSOR {{sptr_name}};
{% endif %}
{
    TensorIdxImporter t_import;
    {% if ref_count %}
    ctx.add(t_import.{{importer_dtype}}_import("{{idx_path}}"),
            "{{tensor_name}}",
            {{ref_count}});
    {% else %}
    ctx.add(t_import.{{importer_dtype}}_import("{{idx_path}}"),
            "{{tensor_name}}");
    {% endif %}
    {% if create_sptr %}
    {{sptr_name}} = ctx.get("{{tensor_name}}");
    {% endif %}
    {% if to_eval %}
    ctx.eval();
    {% endif %}
}