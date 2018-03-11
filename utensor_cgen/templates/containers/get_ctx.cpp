{%if placeholders%}
void get_{{graph_name}}_ctx(Context& ctx, {%for ph in placeholders%}Tensor* input_{{loop.index0}}{%if not loop.last %},{%endif%}{%endfor%}) {

{ // add tensor for placeholders
    {% for ph, ref_count in zip(placeholders, ref_counts) %}
    ctx.add(input_{{loop.index0}}, "{{ph}}", {{ref_count}});
    {% endfor %}
}
{% else %}
void get_{{graph_name}}_ctx(Context& ctx) {
{% endif %}
{% for snippet in snippets%}
{{snippet.render()}}
{% endfor %}
}