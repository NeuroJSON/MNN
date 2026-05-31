//
//  CPUConvolution3D.hpp
//  MNN
//
//  Native 3D convolution for MNN's CPU backend. Without this executor,
//  Conv3D ops in the .mnn fall through GeometryConv3D, which decomposes
//  each Conv3D into an im2col chain whose intermediate tensors balloon
//  to ~400 GB workspace at SIAM v0.3 patch sizes (256x256x192). Direct
//  evaluation of the 3D convolution keeps the workspace at ~6-10 GB.
//
//  Activates only when the runtime gate SIAM_DISABLE_GEOM_CONV3D=1 is
//  set (see source/geometry/GeometryConv3D.cpp). Without the gate,
//  GeometryConv3D continues to claim OpType_Convolution3D and this
//  executor is unreachable.
//
//  MVP: correctness-first 7-loop reference with OpenMP parallelism
//  over the output-channel dimension. No SIMD intrinsics. Fp32 only.
//

#ifndef CPUConvolution3D_hpp
#define CPUConvolution3D_hpp

#include <vector>
#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {

class CPUConvolution3D : public Execution {
public:
    CPUConvolution3D(Backend* backend, const MNN::Op* op);
    virtual ~CPUConvolution3D() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs) override;

private:
    int mKD = 1, mKH = 1, mKW = 1;
    int mSD = 1, mSH = 1, mSW = 1;
    int mPD = 0, mPH = 0, mPW = 0;
    int mDD = 1, mDH = 1, mDW = 1;
    int mInC = 0;
    int mOutC = 0;
    bool mPadModeSame = false;
    bool mRelu  = false;
    bool mRelu6 = false;
    std::vector<float> mWeight;  // [OC, IC, KD, KH, KW] row-major
    std::vector<float> mBias;    // [OC]
};

} // namespace MNN

#endif
