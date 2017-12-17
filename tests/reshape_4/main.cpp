#include "reshape_4_ctx.hpp"
#include "tensorIdxImporter.hpp" 
#include "tensor.hpp"
#include "test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>
#include <math.h>


class ReshapeTest4 : public Test {
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

    ReshapeTest4 test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
    
    return 0;
}

void ReshapeTest4::runAll(void) {
    testStart("simple reshape test 4");
    timer_start();
    get_test_quant_reshape_4_ctx(ctx);

    S_TENSOR out_x = ctx.get("output_x:0");
    ctx.eval();
    timer_stop();

    Tensor* ref_x = t_import.float_import("/fs/idx_data/output_x.idx");
    float err = meanAbsErr<float>(ref_x, out_x.get());
    printf("err: %f (< 0.0001)\n", err);
    passed(err < 0.0001);
}