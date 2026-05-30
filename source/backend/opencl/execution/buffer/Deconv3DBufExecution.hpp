//
//  Deconv3DBufExecution.hpp
//  MNN
//
//  Native 3D transposed convolution for MNN's OpenCL buffer backend.
//
//  Without this, OpType_ConvTranspose3D ops on the OpenCL BUFFER
//  backend go through GeometryConvTranspose3D
//  (source/geometry/GeometryConv3D.cpp), which decomposes the op into
//  a virtual (1, batch*oc*kd*kh*kw, batch*oc*od*oh*ow) tensor + Reduce
//  + Bias-Add + Activate chain. At SIAM v0.3 decoder scale
//  (oc=256, k=3, od*oh*ow ~= 9216) the Reduce step silently produces
//  zero or kernel-faults on NVIDIA OpenCL, blocking the 5 post-Concat
//  decoder Conv3D ops in the network.
//
//  This Execution mirrors Conv3DBufExecution: weights pack to
//  [Cout_block, Kd, Kh, Kw, Cin_block, 4_cin, 4_cout] at session-init,
//  then a single kernel-launch per ConvTranspose3D op produces the
//  full output via a GATHER formulation (no atomics; each output
//  voxel finds its own contributing input voxels via
//  id = (od + pd - kd) / sd, conditional on divisibility + bounds).
//
//  Registering this Execution causes MNN's runtime op-creator path to
//  pick us up for OpType_ConvTranspose3D when the .mnn keeps the 3D op
//  intact (i.e. MNN_CONV3D_TURN2D=0 / MNN_DECONV3D_TURN2D=0 at
//  converter time). Set env var SIAM_DISABLE_GEOM_DECONV3D=1 at runtime
//  to additionally skip MNN's GeometryConvTranspose3D registration so
//  the runtime falls through to our native creator without fighting
//  the geometry computer.
//
//  Supports the SIAM v0.3 ResEnc-UNet's full ConvTranspose3D shape set:
//      kernel 3x3x3   stride (2,2,2)   pad (1,1,1)   output_pad (1,1,1)
//  All group=1, dilation=1. Six ops total; channel pairs
//      320->320, 320->320, 320->256, 256->128, 128->64, 64->32.

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef Deconv3DBufExecution_hpp
#define Deconv3DBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class Deconv3DBufExecution : public CommonExecution {
public:
    Deconv3DBufExecution(const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs,
                         const MNN::Op* op,
                         Backend* backend);
    virtual ~Deconv3DBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs) override;

private:
    OpenCLBackend* mOpenCLBackend = nullptr;

    // Convolution3DCommon parameters (cached from the op at ctor time).
    int mKernelD = 0, mKernelH = 0, mKernelW = 0;
    int mStrideD = 0, mStrideH = 0, mStrideW = 0;
    int mPadD    = 0, mPadH    = 0, mPadW    = 0;
    int mOutputChannels = 0;
    int mInputChannels  = 0;

    // GPU-side weight + bias buffers, populated at construction time.
    // Weights are stored in the same layout as Conv3DBufExecution:
    //   [Cout_block, Kd, Kh, Kw, Cin_block, 4_cin, 4_cout]
    // but with the ONNX (Cin, Cout, Kd, Kh, Kw) source addressing
    // (vs Conv3D's (Cout, Cin, Kd, Kh, Kw)) — see the host packer.
    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Buffer> mBiasBuffer;

    std::shared_ptr<KernelWrap> mKernelNC4  = nullptr;
    std::shared_ptr<KernelWrap> mKernelNCHW = nullptr;
    uint32_t mMaxWGS_NC4  = 1;
    uint32_t mMaxWGS_NCHW = 1;

    // Per-launch global / local work sizes resolved in onEncode().
    std::vector<uint32_t> mGWS{1, 1, 1};
    std::vector<uint32_t> mLWS{1, 1, 1};
    uint32_t mMaxWorkGroupSize = 1;
};

} // namespace OpenCL
} // namespace MNN
#endif /* Deconv3DBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
