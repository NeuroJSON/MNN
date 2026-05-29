"""Bisect models: gradually add SIAM features to find what breaks
the n3d (Conv3D-kept-as-5D) path.

Test sequence:
  T1 = tiny:           Conv3D -> IN -> ReLU -> Conv3D -> IN -> ReLU -> Conv3D_s2  (already works)
  T2 = T1 + skip-cat:  add Concat between input and a later layer
  T3 = T1 + deconv:    add ConvTranspose3D (decoder)
  T4 = T1 + pool:      add Pool3D
  T5 = mini U-Net:     full down-up with skip
"""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "T2"

D = H = W = 16

def cw(name, ci, co, k):
    return numpy_helper.from_array(
        (np.random.RandomState(hash(name) % 2**32).randn(co, ci, k, k, k) * 0.01).astype(np.float32),
        name=name)
def dw(name, ci, co, k):  # ConvTranspose3D weights are (cin, cout, k, k, k)
    return numpy_helper.from_array(
        (np.random.RandomState(hash(name) % 2**32).randn(ci, co, k, k, k) * 0.01).astype(np.float32),
        name=name)
def cb(name, c): return numpy_helper.from_array(np.zeros(c, dtype=np.float32), name=name)
def g(name, c):  return numpy_helper.from_array(np.ones(c, dtype=np.float32), name=name)
def b(name, c):  return numpy_helper.from_array(np.zeros(c, dtype=np.float32), name=name)


# Channel counts
C1, C2 = 8, 16

inits = [
    cw("c1_w", 1, C1, 3),  cb("c1_b", C1),  g("ln1_g", C1), b("ln1_b", C1),
    cw("c2_w", C1, C1, 3), cb("c2_b", C1),  g("ln2_g", C1), b("ln2_b", C1),
    cw("c3_w", C1, C2, 3), cb("c3_b", C2),
]

nodes_common = [
    helper.make_node("Conv", ["input","c1_w","c1_b"], ["c1"],
                     kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
    helper.make_node("InstanceNormalization", ["c1","ln1_g","ln1_b"], ["n1"]),
    helper.make_node("Relu", ["n1"], ["r1"]),
    helper.make_node("Conv", ["r1","c2_w","c2_b"], ["c2"],
                     kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
    helper.make_node("InstanceNormalization", ["c2","ln2_g","ln2_b"], ["n2"]),
    helper.make_node("Relu", ["n2"], ["r2"]),
]

if VARIANT == "T2":  # tiny + skip-cat
    # Concat r1 and r2 along channel axis (both have C1=8 channels at the same resolution)
    inits.append(cw("c3_w_cat", C1*2, C2, 3))
    inits.append(cb("c3_b", C2))
    nodes = nodes_common + [
        helper.make_node("Concat", ["r1", "r2"], ["cat"], axis=1),
        helper.make_node("Conv", ["cat","c3_w_cat","c3_b"], ["output"],
                         kernel_shape=[3,3,3], strides=[2,2,2], pads=[1,1,1,1,1,1]),
    ]
    out_shape = [1, C2, D//2, H//2, W//2]
    # Remove the duplicate c3_w/c3_b from inits (we redefined c3_w_cat above)
    inits = [i for i in inits if i.name not in ("c3_w",)]

elif VARIANT == "T3":  # tiny + Deconv (ConvTranspose3D)
    inits.append(dw("ct_w", C2, C1, 3))
    inits.append(cb("ct_b", C1))
    # c1 -> r1 -> c2 -> r2 -> c3(stride2) -> deconv(back to size, C1 channels) -> output
    nodes = nodes_common + [
        helper.make_node("Conv", ["r2","c3_w","c3_b"], ["c3"],
                         kernel_shape=[3,3,3], strides=[2,2,2], pads=[1,1,1,1,1,1]),
        helper.make_node("ConvTranspose", ["c3","ct_w","ct_b"], ["output"],
                         kernel_shape=[3,3,3], strides=[2,2,2], pads=[1,1,1,1,1,1],
                         output_padding=[1,1,1]),
    ]
    out_shape = [1, C1, D, H, W]

elif VARIANT == "T4":  # tiny + Pool3D
    nodes = nodes_common + [
        helper.make_node("MaxPool", ["r2"], ["pooled"],
                         kernel_shape=[2,2,2], strides=[2,2,2]),
        helper.make_node("Conv", ["pooled","c3_w","c3_b"], ["output"],
                         kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
    ]
    out_shape = [1, C2, D//2, H//2, W//2]

elif VARIANT == "T5":  # mini U-Net: down-up with skip-cat
    inits.append(dw("ct_w", C2, C1, 3))
    inits.append(cb("ct_b", C1))
    inits.append(cw("c4_w", C1*2, C1, 3))   # after concat with skip
    inits.append(cb("c4_b", C1))
    nodes = nodes_common + [
        # downsample
        helper.make_node("Conv", ["r2","c3_w","c3_b"], ["enc_down"],
                         kernel_shape=[3,3,3], strides=[2,2,2], pads=[1,1,1,1,1,1]),
        # upsample
        helper.make_node("ConvTranspose", ["enc_down","ct_w","ct_b"], ["dec_up"],
                         kernel_shape=[3,3,3], strides=[2,2,2], pads=[1,1,1,1,1,1],
                         output_padding=[1,1,1]),
        # skip-cat r2 (encoder feature) and dec_up
        helper.make_node("Concat", ["r2","dec_up"], ["cat"], axis=1),
        # final conv
        helper.make_node("Conv", ["cat","c4_w","c4_b"], ["output"],
                         kernel_shape=[3,3,3], strides=[1,1,1], pads=[1,1,1,1,1,1]),
    ]
    out_shape = [1, C1, D, H, W]

else:
    print(f"unknown variant {VARIANT}", file=sys.stderr); sys.exit(1)

input_info  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, 1, D, H, W])
output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)
graph = helper.make_graph(nodes, VARIANT, [input_info], [output_info], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
out_path = f"/tmp/bisect_{VARIANT}.onnx"
onnx.save(model, out_path)
print(f"saved {out_path} (out_shape={out_shape})")
