#include "conv_ctx.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "TESTS/test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

class ConvTest : public Test {
    Context ctx;
    TensorIdxImporter t_import;
public:
    void runAll(void);
};

int main(int argc, char* argv[]) {

    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

    ConvTest test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

    return 0;
}

void ConvTest::runAll(void) {
    testStart("simple conv test");
    timer_start();
    get_test_quant_conv_ctx(ctx);
    S_TENSOR out = ctx.get("out_conv:0");
    Tensor* ptr_out = out.get();
    ctx.eval();
    timer_stop();

    Tensor* ref_out = t_import.float_import("/fs/idx_data/output_conv.idx");
    double err = meanPercentErr<float>(ref_out, ptr_out);
    printf("%f\n", err);
    passed(err < 0.03);
}