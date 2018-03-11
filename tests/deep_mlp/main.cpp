#include "quant_mnist.hpp"
#include "tensorIdxImporter.hpp"
#include "tensor.hpp"
#include "mbed.h"
#include "emscripten.h"
#include "C12832.h"
#include <string>


C12832 lcd(SPI_MOSI, SPI_SCK, SPI_MISO, p8, p11);
EventQueue queue;
InterruptIn btn(BUTTON1);

void run_mlp(){
    EM_ASM({
        // this writes the content of the canvas (in the simulator) to /fs/tmp.idx
        window.dumpCanvasToTmpFile();
    });

    TensorIdxImporter t_import;
    Tensor* input_x = t_import.float_import("/fs/tmp.idx");
    Context ctx;

    get_quant_mnist_ctx(ctx, input_x);
    S_TENSOR pred_tensor = ctx.get("y_pred:0");
    ctx.eval();

    int pred_label = *(pred_tensor->read<int>(0, 0));
    lcd.cls();
    lcd.locate(3, 13);
    lcd.printf("Predicted label: %d", pred_label);
}

int main(void){
    init_env();
    printf("Simple MNIST end-to-end uTensor cli example on mbed-simulator\n");
    
    btn.fall(queue.event(&run_mlp));
    queue.dispatch_forever();

    return 0;
}
