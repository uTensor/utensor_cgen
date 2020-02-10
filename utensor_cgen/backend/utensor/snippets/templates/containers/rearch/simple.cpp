using namespace uTensor;

{%for snippet in declare_snippets%}
{{snippet.render()}}
{%endfor%}
localCircularArenaAllocator<{{meta_data_pool_size}}> meta_allocator;
localCircularArenaAllocator<{{ram_data_pool_size}}> ram_allocator;

void compute_{{model_name}}({%for pl in placeholders%}Tensor& {{pl}}, {%endfor%}std::vector<Tensor>& outputs){
    Context::get_default_context()->set_metadata_allocator(&meta_allocator);
    Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

    {%for snippet in eval_snippets%}
    {{snippet.render()}}

    {%endfor%}
    {%for out_var in out_tensor_var_names%}
    outputs.push_back({{out_var}});
    {%endfor%}
}
