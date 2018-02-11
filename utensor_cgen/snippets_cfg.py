# -*- coding:utf8 -*-
SNIPPET_CONFIG = {
  "snippets/create_tensor_idx.cpp": set(['"uTensor/loaders/tensorIdxImporter.hpp"', 
                                         '"uTensor/core/context.hpp"', 
                                         '"uTensor/core/tensor.hpp"']),
  "snippets/create_tensor_new.cpp": set(['"uTensor/core/context.hpp"', '"uTensor/core/tensor.hpp"']),
  "snippets/add_op.cpp": set(['"uTensor/ops/MathOps.hpp"']),
  "snippets/min_op.cpp": set(['"uTensor/ops/MathOps.hpp"']),
  "snippets/max_op.cpp": set(['"uTensor/ops/MathOps.hpp"']),
  "snippets/argmax_op.cpp": set(['"uTensor/ops/MathOps.hpp"']),
  "snippets/dequantize_op.cpp": set(['"uTensor/ops/ArrayOps.hpp"']),
  "snippets/qmatmul_op.cpp": set(['"uTensor/ops/MatrixOps.hpp"']),
  "snippets/quantV2_op.cpp": set(['"uTensor/ops/ArrayOps.hpp"']),
  "snippets/qrelu_op.cpp": set(['"uTensor/ops/NnOps.hpp"']),
  "snippets/reshape_op.cpp": set(['"uTensor/ops/ArrayOps.hpp"']),
  "snippets/requant_op.cpp": set(['"uTensor/ops/MathOps.hpp"']),
  "snippets/requant_range_op.cpp": set(['"uTensor/ops/MathOps.hpp"']),
  "snippets/comments.cpp": set([]),
  "snippets/get_ctx.hpp": set(['"uTensor/core/context.hpp"', '"uTensor/core/tensor.hpp"']),
  "containers/main.cpp": set([]),
  "containers/get_ctx.cpp": set([])
}
