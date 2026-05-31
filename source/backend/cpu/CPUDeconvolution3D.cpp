//
//  CPUDeconvolution3D.cpp
//  MNN
//
//  See header.
//

#include "backend/cpu/CPUDeconvolution3D.hpp"

#include <algorithm>
#include <cstdint>

#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUDeconvolution3D::CPUDeconvolution3D(Backend* backend, const MNN::Op* op)
    : Execution(backend) {
    // ConvTranspose3D shares Op storage with Convolution3D.
    const auto* conv3d = op->main_as_Convolution3D();
    MNN_ASSERT(conv3d != nullptr);
    const auto* common = conv3d->common();

    mKD = common->kernels()->Get(0);
    mKH = common->kernels()->Get(1);
    mKW = common->kernels()->Get(2);
    mSD = common->strides()->Get(0);
    mSH = common->strides()->Get(1);
    mSW = common->strides()->Get(2);
    if (common->pads() != nullptr && common->pads()->size() >= 3) {
        mPD = common->pads()->Get(0);
        mPH = common->pads()->Get(1);
        mPW = common->pads()->Get(2);
    }
    if (common->dilates() != nullptr && common->dilates()->size() >= 3) {
        mDD = common->dilates()->Get(0);
        mDH = common->dilates()->Get(1);
        mDW = common->dilates()->Get(2);
    }
    mPadModeSame = (common->padMode() == MNN::PadMode_SAME);
    mInC  = common->inputCount();
    mOutC = common->outputCount();
    mRelu  = common->relu();
    mRelu6 = common->relu6();

    if (conv3d->weight() != nullptr) {
        const float* w = conv3d->weight()->data();
        size_t n = static_cast<size_t>(conv3d->weight()->size());
        mWeight.assign(w, w + n);
    }
    if (conv3d->bias() != nullptr && conv3d->bias()->size() > 0) {
        const float* b = conv3d->bias()->data();
        size_t n = static_cast<size_t>(conv3d->bias()->size());
        mBias.assign(b, b + n);
    } else {
        mBias.assign(static_cast<size_t>(mOutC), 0.0f);
    }
}

ErrorCode CPUDeconvolution3D::onResize(const std::vector<Tensor*>& /*inputs*/,
                                       const std::vector<Tensor*>& /*outputs*/) {
    return NO_ERROR;
}

static inline float fuse_act_(float v, bool relu, bool relu6) {
    if (relu6) {
        if (v < 0.0f) return 0.0f;
        if (v > 6.0f) return 6.0f;
        return v;
    }
    if (relu) {
        return v < 0.0f ? 0.0f : v;
    }
    return v;
}

ErrorCode CPUDeconvolution3D::onExecute(const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs) {
    const Tensor* in = inputs[0];
    Tensor*       out = outputs[0];

    const int N  = in->length(0);
    const int IC = in->length(1);
    const int ID = in->length(2);
    const int IH = in->length(3);
    const int IW = in->length(4);

    const int OC = out->length(1);
    const int OD = out->length(2);
    const int OH = out->length(3);
    const int OW = out->length(4);

    const int KD = mKD, KH = mKH, KW = mKW;
    const int SD = mSD, SH = mSH, SW = mSW;
    const int DD = mDD, DH = mDH, DW = mDW;

    // For ConvTranspose, MNN's pad fields describe trimming applied to the
    // upsampled output; SAME mode bakes equivalent symmetric pads.
    int padFrontD = mPD, padTopH = mPH, padLeftW = mPW;
    if (mPadModeSame) {
        const int needD = std::max(0, ID * SD - OD + (KD - 1) * DD);
        const int needH = std::max(0, IH * SH - OH + (KH - 1) * DH);
        const int needW = std::max(0, IW * SW - OW + (KW - 1) * DW);
        padFrontD = needD / 2;
        padTopH   = needH / 2;
        padLeftW  = needW / 2;
    }

    const float* inputData  = in->host<float>();
    float*       outputData = out->host<float>();
    const float* weightData = mWeight.data();
    const float* biasData   = mBias.data();
    const bool   relu  = mRelu;
    const bool   relu6 = mRelu6;

    const std::int64_t inStrideC  = static_cast<std::int64_t>(ID) * IH * IW;
    const std::int64_t inStrideN  = static_cast<std::int64_t>(IC) * inStrideC;
    const std::int64_t outStrideC = static_cast<std::int64_t>(OD) * OH * OW;
    const std::int64_t outStrideN = static_cast<std::int64_t>(OC) * outStrideC;
    // ConvTranspose weight layout: [IC, OC, KD, KH, KW]
    const std::int64_t wStrideIC  = static_cast<std::int64_t>(OC) * KD * KH * KW;
    const std::int64_t wStrideOC  = static_cast<std::int64_t>(KD) * KH * KW;
    const std::int64_t wStrideKD  = static_cast<std::int64_t>(KH) * KW;

    auto cpuBackend = static_cast<CPUBackend*>(backend());
    const int threadNumber = std::max(1, cpuBackend->threadNumber());

    // Gather formulation: for each output element, sum contributions from
    // every input element that maps to it through the transposed kernel.
    //
    //   od = id * SD - padFrontD + kd * DD
    //   id = (od + padFrontD - kd * DD) / SD   (must be exact and in range)
    //
    for (int n = 0; n < N; ++n) {
        const float* in_n  = inputData  + n * inStrideN;
        float*       out_n = outputData + n * outStrideN;

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            for (int oc = static_cast<int>(tId); oc < OC; oc += threadNumber) {
                float*      out_oc = out_n + oc * outStrideC;
                const float bias   = biasData[oc];

                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        float* out_row = out_oc + (static_cast<std::int64_t>(od) * OH + oh) * OW;
                        for (int ow = 0; ow < OW; ++ow) {
                            float acc = bias;
                            for (int ic = 0; ic < IC; ++ic) {
                                const float* in_ic = in_n + ic * inStrideC;
                                const float* w_ic  = weightData + ic * wStrideIC + oc * wStrideOC;
                                for (int kd = 0; kd < KD; ++kd) {
                                    const int idNum = od + padFrontD - kd * DD;
                                    if (idNum < 0 || (idNum % SD) != 0) continue;
                                    const int id = idNum / SD;
                                    if (id >= ID) continue;
                                    for (int kh = 0; kh < KH; ++kh) {
                                        const int ihNum = oh + padTopH - kh * DH;
                                        if (ihNum < 0 || (ihNum % SH) != 0) continue;
                                        const int ih = ihNum / SH;
                                        if (ih >= IH) continue;
                                        const float* in_row = in_ic + (static_cast<std::int64_t>(id) * IH + ih) * IW;
                                        const float* w_row  = w_ic + kd * wStrideKD + kh * KW;
                                        for (int kw = 0; kw < KW; ++kw) {
                                            const int iwNum = ow + padLeftW - kw * DW;
                                            if (iwNum < 0 || (iwNum % SW) != 0) continue;
                                            const int iw = iwNum / SW;
                                            if (iw >= IW) continue;
                                            acc += in_row[iw] * w_row[kw];
                                        }
                                    }
                                }
                            }
                            out_row[ow] = fuse_act_(acc, relu, relu6);
                        }
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class CPUDeconvolution3DCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& /*inputs*/,
                                const std::vector<Tensor*>& /*outputs*/,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUDeconvolution3D(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDeconvolution3DCreator, OpType_ConvTranspose3D);

} // namespace MNN
