#include <stdint.h>

{% for snippet in snippets%}
{{snippet.render()}}
{% endfor %}

