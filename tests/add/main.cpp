#include "add_ctx.hpp"
#include "context.hpp"
#include "tensorIdxImporter.hpp"

int main(int argc, char* argv[]) {

    Context ctx;
    get_test_quant_add_ctx(ctx);
    Tensor* z = ctx.get("z:0").get();

    TensorIdxImporter t_import;
    Tensor* ref_z = t_import.float_import("/fs/idx_data/output_z.idx");

    // compare the results

    return 0;
}