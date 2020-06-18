#include <stdint.h>

{% for snippet in snippets%}
{{snippet.render()}}
{%if not loop.last%}

{%endif%}
{% endfor %}

