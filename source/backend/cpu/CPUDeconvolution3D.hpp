//
//  CPUDeconvolution3D.hpp
//  MNN
//
//  Native 3D transposed convolution for MNN's CPU backend. Paired with
//  CPUConvolution3D (see that header). Activates when the runtime gate
//  SIAM_DISABLE_GEOM_DECONV3D=1 is set, which suppresses MNN's
//  GeometryConvTranspose3D decomposition.
//
//  MVP: gather-formulation 7-loop reference with OpenMP parallelism
//  over the output-channel dimension. Fp32 only.
//

#ifndef CPUDeconvolution3D_hpp
#define CPUDeconvolution3D_hpp

#include <vector>
#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {

class CPUDeconvolution3D : public Execution {
public:
    CPUDeconvolution3D(Backend* backend, const MNN::Op* op);
    virtual ~CPUDeconvolution3D() = default;
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
    std::vector<float> mWeight;  // ConvTranspose3D weight: [IC, OC, KD, KH, KW]
    std::vector<float> mBias;    // [OC]
};

} // namespace MNN

#endif
