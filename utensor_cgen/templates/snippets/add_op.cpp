ctx.push(new AddOp<{{in_dtype}}, {{out_dtype}}>(),
        {{input_tnames}}, 
        {{output_tname}});