from utensor_cgen.ir.utils import declare_attrib_cls


def test_declare_attribs():
    Attributes = declare_attrib_cls("Attributes", allowed_keys={"name": str, "number": int})
    attributes = Attributes()
    attributes["name"] = "uTensor"
    attributes["number"] = 3
    try:
        # the type of the value is incorrect
        attributes["name"] = 3
    except TypeError:
        pass
    try:
        # this key is not allowed
        attributes["float_number"] = 3.14
    except KeyError:
        pass
    try:
        # the key is not string
        attributes[3] = 1.0
    except TypeError:
        pass
