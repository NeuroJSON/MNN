//
//  CPUConvolution3D.cpp
//  MNN
//
//  See header.
//

#include "backend/cpu/CPUConvolution3D.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUConvolution3D::CPUConvolution3D(Backend* backend, const MNN::Op* op)
    : Execution(backend) {
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

ErrorCode CPUConvolution3D::onResize(const std::vector<Tensor*>& /*inputs*/,
                                     const std::vector<Tensor*>& /*outputs*/) {
    return NO_ERROR;
}

ErrorCode CPUConvolution3D::onExecute(const std::vector<Tensor*>& inputs,
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

    int padFrontD = mPD, padTopH = mPH, padLeftW = mPW;
    if (mPadModeSame) {
        const int needD = std::max(0, (OD - 1) * SD + (KD - 1) * DD + 1 - ID);
        const int needH = std::max(0, (OH - 1) * SH + (KH - 1) * DH + 1 - IH);
        const int needW = std::max(0, (OW - 1) * SW + (KW - 1) * DW + 1 - IW);
        padFrontD = needD / 2;
        padTopH   = needH / 2;
        padLeftW  = needW / 2;
    }

    const float* __restrict__ inputData  = in->host<float>();
    float*       __restrict__ outputData = out->host<float>();
    const float* __restrict__ weightData = mWeight.data();
    const float* __restrict__ biasData   = mBias.data();
    const bool relu  = mRelu;
    const bool relu6 = mRelu6;

    const std::int64_t inStrideC  = static_cast<std::int64_t>(ID) * IH * IW;
    const std::int64_t inStrideN  = static_cast<std::int64_t>(IC) * inStrideC;
    const std::int64_t outStrideC = static_cast<std::int64_t>(OD) * OH * OW;
    const std::int64_t outStrideN = static_cast<std::int64_t>(OC) * outStrideC;
    const std::int64_t wStrideOC  = static_cast<std::int64_t>(IC) * KD * KH * KW;
    const std::int64_t wStrideIC  = static_cast<std::int64_t>(KD) * KH * KW;
    const std::int64_t wStrideKD  = static_cast<std::int64_t>(KH) * KW;

    auto cpuBackend = static_cast<CPUBackend*>(backend());
    const int threadNumber = std::max(1, cpuBackend->threadNumber());

    // Loop structure:
    //
    //   Outer parallel over (oc * od) — gives more work units than just oc
    //   for finer load balancing on hosts with more threads than channels.
    //
    //   Per (oc, od), iterate (oh):
    //     Initialize a per-row accumulator rowAcc[OW] with bias.
    //     For each input channel ic, walk the valid kernel taps (kd, kh, kw)
    //     and add the windowed input * weight to the accumulator.
    //     The INNERMOST loop is over `ow` with stride-1 contiguous input
    //     (when SW=1) — compiler auto-vectorizes the SIMD-FMA there. For
    //     SW>1 the loop is still branch-free; the compiler vectorizes
    //     with strided loads.
    //
    //   Bounds checks on (kd, kh) are hoisted outside the channel loop;
    //   the inner ow loop runs over a pre-computed valid range so no
    //   per-element padding branch remains.

    for (int n = 0; n < N; ++n) {
        const float* __restrict__ in_n  = inputData + n * inStrideN;
        float*       __restrict__ out_n = outputData + n * outStrideN;

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            std::vector<float> rowAcc(static_cast<size_t>(OW), 0.0f);

            const int totalUnits = OC * OD;
            for (int unit = static_cast<int>(tId); unit < totalUnits; unit += threadNumber) {
                const int oc = unit / OD;
                const int od = unit % OD;

                // valid kd range so that id = od*SD - padFrontD + kd*DD lies in [0, ID)
                int kdLo = 0;
                int kdHi = KD;
                {
                    const int id_at_0 = od * SD - padFrontD;
                    while (kdLo < KD && id_at_0 + kdLo * DD < 0) ++kdLo;
                    while (kdHi > kdLo && id_at_0 + (kdHi - 1) * DD >= ID) --kdHi;
                }

                float*       __restrict__ out_oc_od = out_n + oc * outStrideC
                                                     + static_cast<std::int64_t>(od) * OH * OW;
                const float* __restrict__ w_oc = weightData + oc * wStrideOC;
                const float  bias = biasData[oc];

                for (int oh = 0; oh < OH; ++oh) {
                    int khLo = 0;
                    int khHi = KH;
                    {
                        const int ih_at_0 = oh * SH - padTopH;
                        while (khLo < KH && ih_at_0 + khLo * DH < 0) ++khLo;
                        while (khHi > khLo && ih_at_0 + (khHi - 1) * DH >= IH) --khHi;
                    }

                    for (int ow = 0; ow < OW; ++ow) {
                        rowAcc[ow] = bias;
                    }

                    for (int ic = 0; ic < IC; ++ic) {
                        const float* __restrict__ in_ic_n = in_n + ic * inStrideC;
                        const float* __restrict__ w_ic    = w_oc + ic * wStrideIC;

                        for (int kd = kdLo; kd < kdHi; ++kd) {
                            const int id = od * SD - padFrontD + kd * DD;
                            const float* __restrict__ in_d = in_ic_n
                                                             + static_cast<std::int64_t>(id) * IH * IW;
                            const float* __restrict__ w_d  = w_ic + kd * wStrideKD;

                            for (int kh = khLo; kh < khHi; ++kh) {
                                const int ih = oh * SH - padTopH + kh * DH;
                                const float* __restrict__ in_row = in_d
                                                                   + static_cast<std::int64_t>(ih) * IW;
                                const float* __restrict__ w_row  = w_d + kh * KW;

                                for (int kw = 0; kw < KW; ++kw) {
                                    const float w = w_row[kw];

                                    // For each output ow:
                                    //   iw = ow * SW + (kw * DW - padLeftW)
                                    //   valid when iw in [0, IW)
                                    const int iw_off = kw * DW - padLeftW;
                                    int owLo, owHi;
                                    if (SW <= 0) {
                                        owLo = 0;
                                        owHi = 0;
                                    } else {
                                        owLo = (-iw_off + SW - 1) / SW;
                                        if (owLo < 0) owLo = 0;
                                        const int upper = IW - 1 - iw_off;
                                        owHi = (upper < 0) ? 0 : (upper / SW + 1);
                                        if (owHi > OW) owHi = OW;
                                        if (owLo > owHi) owLo = owHi;
                                    }

                                    if (SW == 1) {
                                        // Hot path for stride-1: contiguous load over
                                        // in_row, contiguous store into rowAcc -- compiler
                                        // auto-vectorizes the FMA.
                                        const float* __restrict__ src = in_row + iw_off;
                                        for (int ow = owLo; ow < owHi; ++ow) {
                                            rowAcc[ow] += src[ow] * w;
                                        }
                                    } else {
                                        for (int ow = owLo; ow < owHi; ++ow) {
                                            const int iw = ow * SW + iw_off;
                                            rowAcc[ow] += in_row[iw] * w;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    float* __restrict__ out_row = out_oc_od + static_cast<std::int64_t>(oh) * OW;
                    if (relu6) {
                        for (int ow = 0; ow < OW; ++ow) {
                            float v = rowAcc[ow];
                            if (v < 0.0f) v = 0.0f;
                            else if (v > 6.0f) v = 6.0f;
                            out_row[ow] = v;
                        }
                    } else if (relu) {
                        for (int ow = 0; ow < OW; ++ow) {
                            float v = rowAcc[ow];
                            out_row[ow] = v < 0.0f ? 0.0f : v;
                        }
                    } else {
                        for (int ow = 0; ow < OW; ++ow) {
                            out_row[ow] = rowAcc[ow];
                        }
                    }
                }
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

class CPUConvolution3DCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& /*inputs*/,
                                const std::vector<Tensor*>& /*outputs*/,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUConvolution3D(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(CPUConvolution3DCreator, OpType_Convolution3D);

} // namespace MNN
