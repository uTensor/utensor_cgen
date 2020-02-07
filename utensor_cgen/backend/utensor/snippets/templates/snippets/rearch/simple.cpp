#include "{{header_file}}"
#include "uTensor/tensors/RamTensor.hpp"
#include "uTensor/tensors/RomTensor.hpp"

using namespace uTensor;

{%for op_name, op_type in ops_map.items()%}
{{op_type}} {{op_name}};
{%endfor%}

void compute_{{model_name}}({%for pl in placeholders%}Tensor& {{pl}}, {%endfor%}vector<Tensor>& outputs){

}