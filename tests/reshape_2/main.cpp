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
    get_test_quant_reshape_2_ctx(ctx, 
                                 t_import.float_import("/fs/idx_data/input_x.idx"));
    S_TENSOR input_x = ctx.get("x:0");
    S_TENSOR reshaped_x = ctx.get("y_eightbit_reshape_x:0");
    S_TENSOR quant_x = ctx.get("y_eightbit_quantize_x:0");
    S_TENSOR quant_w = ctx.get("w_quint8_const:0");
    S_TENSOR w_min = ctx.get("w_min:0");
    S_TENSOR w_max = ctx.get("w_max:0");
    S_TENSOR quant_y = ctx.get("y_eightbit_quantized_mat_mul:0");
    S_TENSOR sptr_y = ctx.get("y:0");
    ctx.eval();
    timer_stop();

    input_x->printShape();
    reshaped_x->printShape();
    quant_x->printShape();
    quant_w->printShape();
    w_min->printShape();
    w_max->printShape();
    quant_y->printShape();
    sptr_y->printShape();

    Tensor* ref_y = t_import.float_import("/fs/idx_data/output_y.idx");
    ref_y->setName("ref_y");
    ref_y->printShape();
    float err = meanPercentErr<float>(ref_y, sptr_y.get());
    printf("err: %f (< 0.03)\n", err);
    passed(err < 0.03);
}
