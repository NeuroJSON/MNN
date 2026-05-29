"""Variation tests — change Conv3D #2 to isolate what makes a Conv3D fire vs fail."""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

VARIANT = sys.argv[1]  # "samecout", "smallercout", "stride2", "samecount-1ch"

D = H = W = 16

def conv_w(name, cin, cout, k):
    arr = (np.random.RandomState(hash(name) % 2**32).randn(cout, cin, k, k, k) * 0.01).astype(np.float32)
    return numpy_helper.from_array(arr, name=name)
def conv_b(name, c): return numpy_helper.from_array(np.zeros(c, dtype=np.float32), name=name)
def g(name, c): return numpy_helper.from_array(np.ones(c, dtype=np.float32), name=name)
def b(name, c): return numpy_helper.from_array(np.zeros(c, dtype=np.float32), name=name)

if VARIANT == "samecout":
    c1_cin, c1_cout = 1, 8
    c2_cin, c2_cout, c2_stride = 8, 8, 1
    c3_cin, c3_cout, c3_stride = 8, 16, 2
    out_c, out_sp = 16, D//2
elif VARIANT == "smallercout":  # #2 has different channel count from input
    c1_cin, c1_cout = 1, 8
    c2_cin, c2_cout, c2_stride = 8, 4, 1
    c3_cin, c3_cout, c3_stride = 4, 16, 2
    out_c, out_sp = 16, D//2
elif VARIANT == "stride2":  # #2 is stride-2 instead of stride-1
    c1_cin, c1_cout = 1, 8
    c2_cin, c2_cout, c2_stride = 8, 8, 2
    c3_cin, c3_cout, c3_stride = 8, 16, 1
    out_c, out_sp = 16, D//4
elif VARIANT == "cin4":  # #2 has Cin=4 (cb=1) same shape
    c1_cin, c1_cout = 1, 4
    c2_cin, c2_cout, c2_stride = 4, 4, 1
    c3_cin, c3_cout, c3_stride = 4, 16, 2
    out_c, out_sp = 16, D//2
else:
    print("unknown variant"); sys.exit(1)

inits = [
    conv_w("c1_w", c1_cin, c1_cout, 3), conv_b("c1_b", c1_cout),
    g("ln1_g", c1_cout), b("ln1_b", c1_cout),
    conv_w("c2_w", c2_cin, c2_cout, 3), conv_b("c2_b", c2_cout),
    g("ln2_g", c2_cout), b("ln2_b", c2_cout),
    conv_w("c3_w", c3_cin, c3_cout, 3), conv_b("c3_b", c3_cout),
]

c2_pad = [1,1,1,1,1,1]
nodes = [
    helper.make_node("Conv", ["input","c1_w","c1_b"], ["c1"],
                     kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
    helper.make_node("InstanceNormalization", ["c1","ln1_g","ln1_b"], ["n1"]),
    helper.make_node("Relu", ["n1"], ["r1"]),
    helper.make_node("Conv", ["r1","c2_w","c2_b"], ["c2"],
                     kernel_shape=[3,3,3], strides=[c2_stride]*3, pads=c2_pad),
    helper.make_node("InstanceNormalization", ["c2","ln2_g","ln2_b"], ["n2"]),
    helper.make_node("Relu", ["n2"], ["r2"]),
    helper.make_node("Conv", ["r2","c3_w","c3_b"], ["output"],
                     kernel_shape=[3,3,3], strides=[c3_stride]*3, pads=[1,1,1,1,1,1]),
]

input_info  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, c1_cin, D, H, W])
output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, out_c, out_sp, out_sp, out_sp])

graph = helper.make_graph(nodes, "v", [input_info], [output_info], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.save(model, f"/tmp/tiny_{VARIANT}.onnx")
print(f"saved /tmp/tiny_{VARIANT}.onnx (c1: {c1_cin}->{c1_cout}, c2: {c2_cin}->{c2_cout} s={c2_stride}, c3: {c3_cin}->{c3_cout} s={c3_stride})")
