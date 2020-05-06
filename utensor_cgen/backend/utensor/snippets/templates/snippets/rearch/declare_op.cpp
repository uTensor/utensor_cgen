{%if construct_params%}
{{op_type}} {{op_var_name}}({%for param in construct_params%}{{param}}{%if not loop.last%}, {%endif%}{%endfor%});
{%else%}
{{op_type}} {{op_var_name}};
{%endif%}