using namespace uTensor;

void compute_{{model_name}}({%for pl in placeholders%}Tensor& {{pl}}, {%endfor%}{%for out_tensor in out_tensor_var_names%}Tensor& {{out_tensor}}{%if not loop.last%}, {%endif%}{%endfor%}){
    {#ex: ram tensors#}
    // start rendering local snippets
    {%for snippet in local_snippets%}
    {{snippet.render()}}
    {%if not loop.last%}

    {%endif%}
    {%endfor%}
    // end of rendering local snippets
}
