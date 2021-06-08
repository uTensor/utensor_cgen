{{model_name | model_name_filter}}::{{model_name | model_name_filter}} () :
{%for snp in construct_op_snippets%}
{%if not loop.first%}, {%endif%}{{snp.render()}}{%endfor%}
{
  // Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  // Context::get_default_context()->set_metadata_allocator(&metadata_allocator);
  // TODO: moving ROMTensor declarations here
}

void {{model_name | model_name_filter}}::compute()
{
  // update context in case there are multiple models being run
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_metadata_allocator(&metadata_allocator);
  // start rendering local snippets
  {%for snippet in local_snippets%}
  {{snippet.render()}}
  {%if not loop.last%}

  {%endif%}
  {%endfor%}
  // end of rendering local snippets
}