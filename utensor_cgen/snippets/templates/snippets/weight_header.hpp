#ifndef _{{header_guard}}
#define _{{header_guard}}
{% for val in valarr %}
	{% if val['arr'] %}		
	const {{val['type']}} {{val['name']}}[ {{val['length']}} ] = { {% for v in val['value'] %} {{v[0]}}, {% endfor %} }
	{% else %}
	const {{val['type']}} {{val['name']}} [ {{val['length']}} ] = { {% for v in val['value'] %} {{v}}, {% endfor %} }
	{% endif %}
{% endfor %}
#endif // _{{header_guard}}
