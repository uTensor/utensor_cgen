from utensor_cgen.frontend import FrontendSelector, Parser as _Parser


def test_select_parser():
    FrontendSelector.select_parser('.pb')
    FrontendSelector.select_parser('.onnx')

def test_register():
    class NotParser(object):
        pass

    class Parser(_Parser):
        pass

    try:
        FrontendSelector.register(target_exts=['.test'])(NotParser)
        assert False, "register non parser type"
    except TypeError:
        pass
    FrontendSelector.register(target_exts=['.test'])(Parser)

    try:
        FrontendSelector.register(target_exts=['.test'])(Parser)
        assert False, "duplicate file ext test fail"
    except ValueError:
        pass
