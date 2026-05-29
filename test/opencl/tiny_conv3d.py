"""Build a minimal ONNX model that should reproduce MNN's
"only shape-changing Conv3D fires" failure pattern.

Architecture (mirrors SIAM's per-block structure):
    input  (1, 1, 16, 16, 16)
      |
      Conv3D #1   Cin=1  -> Cout=4    k=3 s=1 p=1     [shape-changing channels: should FIRE]
      LayerNorm
      ReLU
      Conv3D #2   Cin=4  -> Cout=4    k=3 s=1 p=1     [same shape: should FAIL in MNN]
      LayerNorm
      ReLU
      Conv3D #3   Cin=4  -> Cout=8    k=3 s=2 p=1     [shape-changing spatial: should FIRE]
      output

Small spatial (16) and channels (1/4/8) keep VRAM tiny.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

D = H = W = 16
Cin = 1

def conv_w(name, cin, cout, k):
    arr = (np.random.RandomState(hash(name) % 2**32).randn(cout, cin, k, k, k) * 0.01).astype(np.float32)
    return numpy_helper.from_array(arr, name=name)

def conv_b(name, cout):
    arr = np.zeros(cout, dtype=np.float32)
    return numpy_helper.from_array(arr, name=name)

def ln_gamma(name, c):
    return numpy_helper.from_array(np.ones(c, dtype=np.float32), name=name)

def ln_beta(name, c):
    return numpy_helper.from_array(np.zeros(c, dtype=np.float32), name=name)

inits = [
    conv_w("c1_w", 1, 8, 3),
    conv_b("c1_b", 8),
    ln_gamma("ln1_g", 8),
    ln_beta("ln1_b", 8),

    conv_w("c2_w", 8, 8, 3),
    conv_b("c2_b", 8),
    ln_gamma("ln2_g", 8),
    ln_beta("ln2_b", 8),

    conv_w("c3_w", 8, 16, 3),
    conv_b("c3_b", 16),
]

nodes = [
    helper.make_node("Conv", ["input", "c1_w", "c1_b"], ["c1"],
                     kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
    helper.make_node("InstanceNormalization", ["c1", "ln1_g", "ln1_b"], ["n1"]),
    helper.make_node("Relu", ["n1"], ["r1"]),

    helper.make_node("Conv", ["r1", "c2_w", "c2_b"], ["c2"],
                     kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
    helper.make_node("InstanceNormalization", ["c2", "ln2_g", "ln2_b"], ["n2"]),
    helper.make_node("Relu", ["n2"], ["r2"]),

    helper.make_node("Conv", ["r2", "c3_w", "c3_b"], ["output"],
                     kernel_shape=[3,3,3], strides=[2,2,2], pads=[1,1,1,1,1,1]),
]

input_info  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, 1, D, H, W])
output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 16, D//2, H//2, W//2])

graph = helper.make_graph(nodes, "tiny_conv3d", [input_info], [output_info], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "/tmp/tiny_conv3d.onnx")
print("saved /tmp/tiny_conv3d.onnx")
