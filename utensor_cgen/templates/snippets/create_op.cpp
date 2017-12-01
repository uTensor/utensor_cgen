{% if template_type %}
ctx.push(new {{op_name}}(), {{input_tnames}}, {{output_tnames}});
{% else %}
ctx.push(new {{op_name}}<{{template_type}}>(), {{input_tnames}}, {{output_tnames}});
{% endif %}