{%if dtypes %}
{{op_type}}<{%for t in dtypes[:-1]%}{{t}},{%endfor%}{{dtypes[-1]}}> {{op_var_name}};
{%else%}
{{op_type}} {{op_var_name}};
{%endif%}