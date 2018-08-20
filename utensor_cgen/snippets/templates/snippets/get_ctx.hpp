#ifndef _{{header_guard}}
#define _{{header_guard}}
#include "uTensor/core/context.hpp"
{% if placeholders %}
void get_{{graph_name}}_ctx(Context& ctx, {%for ph in placeholders%}Tensor* input_{{loop.index0}}{%if not loop.last %},{%endif%}{%endfor%});
{% else %}
void get_{{graph_name}}_ctx(Context& ctx);
{% endif %}
#endif // _{{header_guard}}
