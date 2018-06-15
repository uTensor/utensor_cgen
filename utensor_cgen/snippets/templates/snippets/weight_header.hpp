{% for val in valarr %}
    {% if val['arr'] %}
		const {{val['type']}} {{val['inline_name']}} [ {{val['length']}} ] = { {% for v in val['value'] %} {{v[0]}}, {% endfor %} };
	{% else %}
		const {{val['type']}} {{val['inline_name']}} [ {{val['length']}} ] = { {% for v in val['value'] %} {{v}}, {% endfor %} };
	{% endif %}
{% endfor %}

