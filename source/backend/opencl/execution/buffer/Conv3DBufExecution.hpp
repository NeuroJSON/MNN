//
//  Conv3DBufExecution.hpp
//  MNN
//
//  Native 3D convolution for MNN's OpenCL buffer backend.
//
//  Without this, MNN's converter rewrites every Conv3D into ~40 Conv2D
//  + Im2Col + Unpack + Transpose ops at conversion time
//  (Convolution3DTurn2D, in tools/converter/source/optimizer/merge/),
//  blowing a single forward pass of SIAM's ResEnc-UNet up to
//  ~3300 small kernel launches and creating a launch-bound GPU
//  profile that wastes ~90 % of GPU time waiting between dispatches.
//  Registering this Execution causes MNN's runtime op-creator path to
//  pick us up for OpType_Convolution3D, BUT for the runtime to even
//  see Conv3D ops, the .mnn must have been produced with
//  `mnnconvert --optimizeLevel 0` (or with the converter patched to
//  skip Convolution3DTurn2D for ops whose target backend has a native
//  Conv3D registered).
//
//  Supports the SIAM v0.3 ResEnc-UNet's full Conv3D shape set:
//      kernel 1x1x1   stride (1,1,1) pad (0,0,0)
//      kernel 3x3x3   stride (1,1,1) pad (1,1,1)
//      kernel 3x3x3   stride (2,2,2) pad (1,1,1)
//      kernel 3x3x3   stride (2,2,1) pad (1,1,1)
//  All group=1, dilation=1.

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef Conv3DBufExecution_hpp
#define Conv3DBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class Conv3DBufExecution : public CommonExecution {
public:
    Conv3DBufExecution(const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs,
                       const MNN::Op* op,
                       Backend* backend);
    virtual ~Conv3DBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs,
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
    // Weights are stored as a plain (Cout, Cin, Kd, Kh, Kw) FLOAT (fp16
    // or fp32 depending on precision) buffer matching the kernel's
    // access pattern.
    std::shared_ptr<cl::Buffer> mKernelBuffer;
    std::shared_ptr<cl::Buffer> mBiasBuffer;

    // Two kernel variants: NC4HW4 reader for standard inputs, NCHW
    // reader for inputs we can take directly from a MEMORY_VIRTUAL
    // Region's NCHW source. Picked per-encode based on input layout.
    std::shared_ptr<KernelWrap> mKernelNC4 = nullptr;
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
#endif /* Conv3DBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
