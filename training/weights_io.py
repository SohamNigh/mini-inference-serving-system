"""MIN1 + three uint32 dimensions + row-major W1,b1,W2,b2 float32; all LE."""
import struct
import numpy as np

def export(path, w1, b1, w2, b2):
    assert w1.shape[1] == w2.shape[0]
    assert b1.shape == (w1.shape[1],) and b2.shape == (w2.shape[1],)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<4sIII", b"MIN1", w1.shape[0], w1.shape[1], w2.shape[1]))
        for array in (w1, b1, w2, b2):
            stream.write(np.asarray(array, dtype="<f4").tobytes(order="C"))
