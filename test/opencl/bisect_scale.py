"""Scale T5 (mini U-Net) to varying sizes to find where it breaks."""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# (D, H, W, C1, C2)
PRESETS = {
    "S16":  (16, 16, 16, 8, 16),     # original tiny
    "S32":  (32, 32, 32, 8, 16),
    "S64":  (64, 64, 64, 16, 32),
    "S96":  (96, 96, 64, 32, 64),
    "S192": (192, 192, 128, 32, 64),
    "S192c": (192, 192, 128, 32, 128),     # SIAM-like channel count
    "S192cc": (192, 192, 128, 32, 320),    # SIAM bottleneck channels
}

KEY = sys.argv[1] if len(sys.argv) > 1 else "S192"
D, H, W, C1, C2 = PRESETS[KEY]

def cw(name, ci, co, k):
    return numpy_helper.from_array(
        (np.random.RandomState(hash(name) % 2**32).randn(co, ci, k, k, k) * 0.01).astype(np.float32),
        name=name)
def dw(name, ci, co, k):
    return numpy_helper.from_array(
        (np.random.RandomState(hash(name) % 2**32).randn(ci, co, k, k, k) * 0.01).astype(np.float32),
        name=name)
def cb(name, c): return numpy_helper.from_array(np.zeros(c, dtype=np.float32), name=name)
def g(name, c):  return numpy_helper.from_array(np.ones(c, dtype=np.float32), name=name)
def b(name, c):  return numpy_helper.from_array(np.zeros(c, dtype=np.float32), name=name)

inits = [
    cw("c1_w", 1, C1, 3),  cb("c1_b", C1),  g("ln1_g", C1), b("ln1_b", C1),
    cw("c2_w", C1, C1, 3), cb("c2_b", C1),  g("ln2_g", C1), b("ln2_b", C1),
    cw("c3_w", C1, C2, 3), cb("c3_b", C2),
    dw("ct_w", C2, C1, 3), cb("ct_b", C1),
    cw("c4_w", C1*2, C1, 3), cb("c4_b", C1),
]

nodes = [
    helper.make_node("Conv", ["input","c1_w","c1_b"], ["c1"],
                     kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
    helper.make_node("InstanceNormalization", ["c1","ln1_g","ln1_b"], ["n1"]),
    helper.make_node("Relu", ["n1"], ["r1"]),
    helper.make_node("Conv", ["r1","c2_w","c2_b"], ["c2"],
                     kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
    helper.make_node("InstanceNormalization", ["c2","ln2_g","ln2_b"], ["n2"]),
    helper.make_node("Relu", ["n2"], ["r2"]),
    helper.make_node("Conv", ["r2","c3_w","c3_b"], ["enc_down"],
                     kernel_shape=[3,3,3], strides=[2,2,2], pads=[1,1,1,1,1,1]),
    helper.make_node("ConvTranspose", ["enc_down","ct_w","ct_b"], ["dec_up"],
                     kernel_shape=[3,3,3], strides=[2,2,2], pads=[1,1,1,1,1,1],
                     output_padding=[1,1,1]),
    helper.make_node("Concat", ["r2","dec_up"], ["cat"], axis=1),
    helper.make_node("Conv", ["cat","c4_w","c4_b"], ["output"],
                     kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
]

iinfo  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, 1, D, H, W])
oinfo  = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C1, D, H, W])
graph  = helper.make_graph(nodes, KEY, [iinfo], [oinfo], inits)
model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
out = f"/tmp/bs_{KEY}.onnx"
onnx.save(model, out)
print(f"saved {out} (D={D} H={H} W={W} C1={C1} C2={C2})")
