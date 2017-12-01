void get_ctx(Context& ctx) {
  {% for snippet in snippets%}
  {{snippet.render()}}
  {% endfor %}
}