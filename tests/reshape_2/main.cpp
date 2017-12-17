#include "reshape_2_ctx.hpp"
#include "tensorIdxImporter.hpp" 
#include "tensor.hpp"
#include "test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>
#include <math.h>


class ReshapeTest2 : public Test {
    Context ctx;
    TensorIdxImporter t_import;
public:
    void runAll(void);
};

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

int main(int argc, char* argv[]) {

    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

    ReshapeTest2 test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
    
    return 0;
}

void ReshapeTest2::runAll(void) {
    testStart("simple reshape test 2");
    timer_start();
    Tensor* input_x = t_import.float_import("/fs/idx_data/input_x.idx");
    get_test_quant_reshape_2_ctx(ctx, input_x);

    S_TENSOR sptr_y = ctx.get("y:0");
    ctx.eval();
    timer_stop();

    Tensor* ref_y = t_import.float_import("/fs/idx_data/output_y.idx");
    float err = meanAbsErr<float>(ref_y, sptr_y.get());
    printf("err: %f (< 0.0001)\n", err);
    passed(err < 0.0001);
}