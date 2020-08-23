#ifndef __{{model_name.upper()}}_INTERFACE_H
#define __{{model_name.upper()}}_INTERFACE_H
#include "uTensor.h"

using namespace uTensor;

class {{model_name | model_name_filter}} : public ModelInterface<{{len(placeholders)}}, {{len(out_tensor_var_names)}}> 
{
 public:
 {%if input_enums %}
  enum input_names : uint8_t { {%for in_enum in input_enums%}{{in_enum}}{%if not loop.last%}, {%endif%}{%endfor%} };
 {%endif%}
 {%if output_enums %}
  enum output_names : uint8_t { {%for out_enum in output_enums%}{{out_enum}}{%if not loop.last%}, {%endif%}{%endfor%} };
 {%endif%}
  {{model_name | model_name_filter}}();
 protected:
  virtual void compute();
 private:
  // Operators
{%for op_snp in declare_op_snippets %}
  {{op_snp.render()}}
{%endfor%}
  // memory allocators
  localCircularArenaAllocator<{{ram_data_pool_size}}, {{ram_dtype}}> ram_allocator;
  localCircularArenaAllocator<{{meta_data_pool_size}}, {{meta_dtype}}> metadata_allocator;
};

#endif // __{{model_name.upper()}}_INTERFACE_H