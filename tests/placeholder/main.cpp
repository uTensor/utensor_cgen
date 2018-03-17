#include "placeholder_ctx.hpp"
#include "uTensor/core/tensor.hpp"
#include "uTensor/loaders/tensorIdxImporter.hpp"
#include "TESTS/test.hpp"
#include <mbed.h>
#include <FATFileSystem.h>
#include <SDBlockDevice.h>
#include <math.h>


class PlaceholderTest : public Test {
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

    PlaceholderTest test;
    test.runAll();
    test.printSummary();

    ON_ERR(fs.unmount(), "fs unmount ");
    ON_ERR(bd.deinit(), "SDBlockDevice de-init ");
    
    return 0;
}

void PlaceholderTest::runAll(void) {
    testStart("simple placeholder test");
    timer_start();
    Tensor* input_0 = new RamTensor<float>({1});
    *(input_0->write<float>(0, 0)) = 3.1415;
    get_test_quant_placeholder_ctx(ctx, input_0);

    S_TENSOR sptr_y = ctx.get("y:0");
    ctx.eval();
    timer_stop();

    float ans = 4.1415, output = *(sptr_y->read<float>(0, 0));
    passed(fabs(ans - output) < 0.0001);
}