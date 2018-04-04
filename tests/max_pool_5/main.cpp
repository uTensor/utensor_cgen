#include "max_pool_5_ctx.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "TESTS/test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>


class MaxPoolTest5 : public Test {
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

    MaxPoolTest5 test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

    return 0;
}

void MaxPoolTest5::runAll(void) {
    testStart("simple max_pool_5 test");
    timer_start();
    get_test_quant_max_pool_5_ctx(ctx);
    S_TENSOR pool5 = ctx.get("pool5:0");
    ctx.eval();
    timer_stop();

    Tensor* ref_pool5 = t_import.float_import("/fs/idx_data/output_max_pool_5.idx");

    // compare the results
    double err = meanPercentErr<float>(ref_pool5, pool5.get());
    printf("meanPercentErr: %f (<0.03?)\n", err);
    passed(err < 0.03);
}
