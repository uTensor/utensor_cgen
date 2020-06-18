#ifndef __{{model_name.upper()}}_H
#define __{{model_name.upper()}}_H

#include "uTensor/core/tensor.hpp"

constexpr {{ram_dtype}} estimated_ram_usage = {{ram_data_pool_size}};
constexpr {{meta_dtype}} estimated_meta_usage = {{meta_data_pool_size}};

void compute_{{model_name}}({%for pl in placeholders%}uTensor::Tensor& {{pl}}, {%endfor%}{%for out_tensor in out_tensor_var_names%}uTensor::Tensor& {{out_tensor}}{%if not loop.last%}, {%endif%}{%endfor%});

#endif // __{{model_name.upper()}}_H