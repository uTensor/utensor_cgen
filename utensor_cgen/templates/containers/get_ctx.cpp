void get_{{graph_name}}_ctx(Context& ctx) {
{% for snippet in snippets%}
{{snippet.render()}}
{% endfor %}
}