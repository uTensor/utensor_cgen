#ifndef __{{model_name.upper()}}_H
#define __{{model_name.upper()}}_H

#include <vector>
#include "uTensor/core/tensor.hpp"

void compute_{{model_name}}({%for pl in placeholders%}uTensor::Tensor& {{pl}}, {%endfor%}std::vector<uTensor::Tensor>& outputs);

#endif // __{{model_name.upper()}}_H