{%if construct_params %}
{{op_var_name}}({%for param in construct_params%}{{param}}{%if not loop.last%}, {%endif%}{%endfor%})
{%else%}
{{op_var_name}}()
{%endif%}