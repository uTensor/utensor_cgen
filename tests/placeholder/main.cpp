#include "placeholder_ctx.hpp"
#include "tensorIdxImporter.hpp" 
#include "tensor.hpp"
#include <stdio.h>


int main(int argc, char* argv[]) {
    Context ctx;
    TensorIdxImporter t_import;
    Tensor* input_0 = new RamTensor<float>({1});
    *(input_0->write<float>(0, 0)) = 3.1415;
    get_test_quant_placeholder_ctx(ctx, input_0);

    S_TENSOR sptr_y = ctx.get("y:0");
    ctx.eval();

    printf("get %f, expect %f", *(sptr_y->read<float>(0, 0)), 4.1415);
    return 0;
}
