#include "conv_ctx.hpp"
#include "tensor.hpp"
#include "tensorIdxImporter.hpp"

int main(int argc, char* argv[]) {
    Context ctx;
    get_test_quant_conv_ctx(ctx);
    S_TENSOR output = ctx.get("out_conv:0");
    ctx.eval();

    return 0;
}