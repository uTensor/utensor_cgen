# -*- coding:utf8 -*-
SNIPPET_CONFIG = {
  "hello_world.cpp": set(["<stdio.h>"]),
  "create_tensor_idx.cpp": set(['"tensorIdxImporter.hpp"', '"context.hpp"', '"tensor.hpp"']),
  "create_tensor_new.cpp": set(['"context.hpp"', '"tensor.hpp"']),
  "create_op.cpp": set(["context.hpp"]),
  "headers.hpp": set([]),
  "get_ctx.hpp": set([])
}
CONTAINER_CONFIG = {
  "main.cpp": set([]),
  "get_ctx.cpp": set(['"get_ctx.hpp"'])
}
