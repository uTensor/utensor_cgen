#include "add_ctx.hpp"
#include "tensorIdxImporter.hpp"
#include "uTensor_util.hpp"
#include "test.hpp"


class AddTest : public Test {
    Context ctx;
    TensorIdxImporter t_import;
public:
    void runAll(void) {
        testStart("simple add test");
        timer_start();
        get_test_quant_add_ctx(ctx);
        S_TENSOR z = ctx.get("z:0");
        Tensor* ptr_z = z.get();
        ctx.eval();
        timer_stop();

        Tensor* ref_z = t_import.float_import("/fs/idx_data/output_z.idx");
    
        // compare the results
        if (ptr_z->getSize() != ref_z->getSize()) {
            ERR_EXIT("size mismatch");
        }

        double result = meanPercentErr<float>(ref_z, ptr_z);
        passed(result < 0.001);
    }
};

int main(int argc, char* argv[]) {

    AddTest test;
    test.runAll();

    return 0;
}