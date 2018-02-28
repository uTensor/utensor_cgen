#include "max_1_1_ctx.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "TESTS/test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>


class MaxTest : public Test {
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

    MaxTest test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

    return 0;
}

void MaxTest::runAll(void) {
    testStart("simple max test 1_1");
    timer_start();
    get_test_quant_max_1_1_ctx(ctx);
    S_TENSOR max_x1 = ctx.get("max_x1_1:0");
    ctx.eval();
    timer_stop();

    Tensor* ref_max1 = t_import.float_import("/fs/idx_data/output_max_x1_1.idx");

    // compare the results
    double err = meanAbsErr<float>(ref_max1, max_x1.get());
    printf("err: %f\n", err);
    passed(err < 0.0003);
}
