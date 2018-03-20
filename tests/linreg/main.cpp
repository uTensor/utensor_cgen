#include "linreg_ctx.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "uTensor/util/uTensor_util.hpp"
#include "TESTS/test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>


class LinregTest : public Test {
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

    LinregTest test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

    return 0;
}

void LinregTest::runAll(void) {
    testStart("simple linreg matmul test");
    timer_start();
    get_test_quant_linreg_ctx(ctx);
    S_TENSOR yhat = ctx.get("yhat:0");
    ctx.eval();
    timer_stop();

    Tensor* ref_yhat = t_import.float_import("/fs/idx_data/output_yhat.idx");

    // compare the results
    double percErr = meanPercentErr<float>(ref_yhat, yhat.get());
    printf("percErr: %f (< 0.05)\n", percErr);
    passed(percErr < 0.05);
}
