# -*- coding:utf8 -*-
SNIPPET_CONFIG = {
  "hello_world.cpp": set(["<stdio.h>"]),
  "create_tensor_idx.cpp": set(['"tensorIdxImporter.hpp"', '"context.hpp"', '"tensor.hpp"']),
  "create_tensor_new.cpp": set(['"context.hpp"', '"tensor.hpp"']),
  "add_op.cpp": set(['"MathOps.hpp"']),
  "min_op.cpp": set(['"MathOps.hpp"']),
  "max_op.cpp": set(['"MathOps.hpp"']),
  "argmax_op.cpp": set(['"MathOps.hpp"']),
  "dequantize_op.cpp": set(['"ArrayOps.hpp"']),
  "qmatmul_op.cpp": set(['"MatrixOps.hpp"']),
  "quantV2_op.cpp": set(['"ArrayOps.hpp"']),
  "qrelu_op.cpp": set(['"NnOps.hpp"']),
  "reshape_op.cpp": set(['"ArrayOps.hpp"']),
  "requant_op.cpp": set(['"MathOps.hpp"']),
  "requant_range_op.cpp": set(['"MathOps.hpp"']),
  "comments.cpp": set([]),
  "headers.hpp": set([]),
  "get_ctx.hpp": set(['"context.hpp"', '"tensor.hpp"'])
}

CONTAINER_CONFIG = {
  "main.cpp": set([]),
  "get_ctx.cpp": set([])
}
