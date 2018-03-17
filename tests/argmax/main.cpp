#include "argmax_ctx.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "TESTS/test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>


class ArgmaxTest : public Test {
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

    ArgmaxTest test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

    return 0;
}

void ArgmaxTest::runAll(void) {
    testStart("simple argmax test");
    timer_start();
    get_test_quant_argmax_ctx(ctx);
    S_TENSOR argmax = ctx.get("argmax:0");
    ctx.eval();
    timer_stop();

    Tensor* ref_argmax = t_import.int_import("/fs/idx_data/output_argmax.idx");
    double err = meanAbsErr<float>(ref_argmax, argmax.get());
    passed(err == 0);
}