#include <stdint.h>

const {{ type }} {{ inline_name }} [ {{ length }} ] = { {% for item in value %} {{ item }}, {% endfor %} };
