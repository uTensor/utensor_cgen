# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers

class GreaterEqualOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsGreaterEqualOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GreaterEqualOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GreaterEqualOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # GreaterEqualOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def GreaterEqualOptionsStart(builder): builder.StartObject(0)
def GreaterEqualOptionsEnd(builder): return builder.EndObject()
