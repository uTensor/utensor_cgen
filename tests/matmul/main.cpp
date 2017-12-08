#include "matmul_ctx.hpp"
#include "tensorIdxImporter.hpp"
#include "uTensor_util.hpp"
#include "test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>


class qMatMulTest : public Test {
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

    qMatMulTest test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

    return 0;
}

void qMatMulTest::runAll(void) {
    testStart("simple quantized matmul test");
    timer_start();
    get_test_quant_matmul_ctx(ctx);
    S_TENSOR z = ctx.get("z:0");
    ctx.eval();
    timer_stop();

    Tensor* ptr_z = z.get();
    Tensor* ref_z = t_import.float_import("/fs/idx_data/output_z.idx");

    // compare the results
    double percErr = meanPercentErr<float>(ref_z, ptr_z);
    printf("percErr: %f (< 0.003)\n", percErr);
    passed(percErr < 0.003);
}
