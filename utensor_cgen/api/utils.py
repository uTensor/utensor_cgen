import textwrap

import click


def show_ugraph(ugraph, oneline=False, ignore_unknown_op=False):
  from utensor_cgen.backend.utensor.code_generator.legacy._operators import OperatorFactory

  unknown_ops = set([])
  if oneline:
    tmpl = click.style("{op_name} ", fg='yellow', bold=True) + \
      "op_type: {op_type}, inputs: {inputs}, outputs: {outputs}"
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      msg = tmpl.format(op_name=op_name, op_type=op_info.op_type,
                        inputs=[tensor.name for tensor in op_info.input_tensors],
                        outputs=[tensor.name for tensor in op_info.output_tensors])
      click.echo(msg)
      if not OperatorFactory.is_supported(op_info.op_type):
        unknown_ops.add(op_info)
  else:
    tmpl = click.style('op_name: {op_name}\n', fg='yellow', bold=True) + \
    '''\
      op_type: {op_type}
      input(s):
        {inputs}
        {input_shapes}
      ouptut(s):
        {outputs}
        {output_shapes}
    '''
    tmpl = textwrap.dedent(tmpl)
    paragraphs = []
    for op_name in ugraph.topo_order:
      op_info = ugraph.ops_info[op_name]
      op_str = tmpl.format(
        op_name=op_name,
        op_type=op_info.op_type,
        inputs=op_info.input_tensors,
        outputs=op_info.output_tensors,
        input_shapes=[tensor.shape for tensor in op_info.input_tensors],
        output_shapes=[tensor.shape for tensor in op_info.output_tensors])
      paragraphs.append(op_str)
      if not OperatorFactory.is_supported(op_info.op_type):
        unknown_ops.add(op_info)
    click.echo('\n'.join(paragraphs))
  click.secho(
    'topological ordered ops: {}'.format(ugraph.topo_order),
    fg='white', bold=True,
  )
  if unknown_ops and not ignore_unknown_op:
    click.echo(
      click.style('Unknown Ops Detected', fg='red', bold=True)
    )
    for op_info in unknown_ops:
      click.echo(
        click.style('    {}: {}'.format(op_info.name, op_info.op_type), fg='red')
      )
  return 0
