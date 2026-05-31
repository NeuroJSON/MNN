//
//  CPUDeconvolution3D.cpp
//  MNN
//
//  See header.
//

#include "backend/cpu/CPUDeconvolution3D.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {

CPUDeconvolution3D::CPUDeconvolution3D(Backend* backend, const MNN::Op* op)
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

ErrorCode CPUDeconvolution3D::onResize(const std::vector<Tensor*>& /*inputs*/,
                                       const std::vector<Tensor*>& /*outputs*/) {
    return NO_ERROR;
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

    int padFrontD = mPD, padTopH = mPH, padLeftW = mPW;
    if (mPadModeSame) {
        const int needD = std::max(0, ID * SD - OD + (KD - 1) * DD);
        const int needH = std::max(0, IH * SH - OH + (KH - 1) * DH);
        const int needW = std::max(0, IW * SW - OW + (KW - 1) * DW);
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
    // ConvTranspose weight layout: [IC, OC, KD, KH, KW]
    const std::int64_t wStrideIC  = static_cast<std::int64_t>(OC) * KD * KH * KW;
    const std::int64_t wStrideOC  = static_cast<std::int64_t>(KD) * KH * KW;
    const std::int64_t wStrideKD  = static_cast<std::int64_t>(KH) * KW;

    auto cpuBackend = static_cast<CPUBackend*>(backend());
    const int threadNumber = std::max(1, cpuBackend->threadNumber());

    // Output-gather formulation:
    //
    //   For each output (n, oc, od, oh):
    //     Initialize rowAcc[OW] with bias.
    //     For each ic, walk the kernel taps (kd, kh, kw) that, given the
    //     transposed-conv index relation
    //         od = id * SD - padFrontD + kd * DD
    //         <=>  id * SD = od + padFrontD - kd * DD
    //     map (od, oh, ow) back to a valid input position (id, ih, iw).
    //     Bounds and divisibility checks (must satisfy idNum % SD == 0)
    //     are hoisted out of the channel loop. Innermost loop is over `ow`
    //     with stride-SW reads from in_row — vectorizable for SW=1 and SW=2.

    for (int n = 0; n < N; ++n) {
        const float* __restrict__ in_n  = inputData + n * inStrideN;
        float*       __restrict__ out_n = outputData + n * outStrideN;

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            std::vector<float> rowAcc(static_cast<size_t>(OW), 0.0f);

            const int totalUnits = OC * OD;
            for (int unit = static_cast<int>(tId); unit < totalUnits; unit += threadNumber) {
                const int oc = unit / OD;
                const int od = unit % OD;

                // Quick-reject bounds on kd: trim leading/trailing values that
                // never satisfy (idNum >= 0 AND idNum < ID*SD AND divisible).
                // idNum = od + padFrontD - kd * DD is MONOTONIC IN kd (decreasing),
                // so the valid kd values form a contiguous-ish band; we still
                // check divisibility in the inner loop because it depends on parity.
                int kdLo = 0;
                int kdHi = KD;
                while (kdLo < kdHi) {
                    const int idNum = od + padFrontD - kdLo * DD;
                    const int id    = (idNum >= 0 && SD > 0 && (idNum % SD) == 0)
                                      ? (idNum / SD) : -1;
                    if (id >= 0 && id < ID) break;
                    ++kdLo;
                }
                while (kdHi > kdLo) {
                    const int kd2   = kdHi - 1;
                    const int idNum = od + padFrontD - kd2 * DD;
                    const int id    = (idNum >= 0 && SD > 0 && (idNum % SD) == 0)
                                      ? (idNum / SD) : -1;
                    if (id >= 0 && id < ID) break;
                    --kdHi;
                }

                float*       __restrict__ out_oc_od = out_n + oc * outStrideC
                                                     + static_cast<std::int64_t>(od) * OH * OW;
                const float  bias = biasData[oc];

                for (int oh = 0; oh < OH; ++oh) {
                    int khLo = 0;
                    int khHi = KH;
                    while (khLo < khHi) {
                        const int ihNum = oh + padTopH - khLo * DH;
                        const int ih = (ihNum >= 0 && (SH > 0) && (ihNum % SH) == 0) ? (ihNum / SH) : -1;
                        if (ih >= 0 && ih < IH) break;
                        ++khLo;
                    }
                    while (khHi > khLo) {
                        const int kh2 = khHi - 1;
                        const int ihNum = oh + padTopH - kh2 * DH;
                        const int ih = (ihNum >= 0 && (SH > 0) && (ihNum % SH) == 0) ? (ihNum / SH) : -1;
                        if (ih >= 0 && ih < IH) break;
                        --khHi;
                    }

                    for (int ow = 0; ow < OW; ++ow) {
                        rowAcc[ow] = bias;
                    }

                    for (int ic = 0; ic < IC; ++ic) {
                        const float* __restrict__ in_ic_n = in_n + ic * inStrideC;
                        const float* __restrict__ w_ic_oc = weightData
                                                             + ic * wStrideIC
                                                             + oc * wStrideOC;

                        for (int kd = kdLo; kd < kdHi; ++kd) {
                            const int idNum = od + padFrontD - kd * DD;
                            if (idNum < 0 || (idNum % SD) != 0) continue;
                            const int id = idNum / SD;
                            if (id < 0 || id >= ID) continue;
                            const float* __restrict__ in_d = in_ic_n
                                                             + static_cast<std::int64_t>(id) * IH * IW;
                            const float* __restrict__ w_d  = w_ic_oc + kd * wStrideKD;

                            for (int kh = khLo; kh < khHi; ++kh) {
                                const int ihNum = oh + padTopH - kh * DH;
                                if (ihNum < 0 || (ihNum % SH) != 0) continue;
                                const int ih = ihNum / SH;
                                if (ih < 0 || ih >= IH) continue;
                                const float* __restrict__ in_row = in_d
                                                                   + static_cast<std::int64_t>(ih) * IW;
                                const float* __restrict__ w_row  = w_d + kh * KW;

                                for (int kw = 0; kw < KW; ++kw) {
                                    const float w = w_row[kw];
                                    // ow valid when (ow + padLeftW - kw*DW) >= 0, divisible by SW,
                                    // and quotient < IW.
                                    const int ow_off = kw * DW - padLeftW;  // ow = iw*SW + ow_off ?
                                    // Actually: iwNum = ow + padLeftW - kw*DW = ow - ow_off
                                    // iw = iwNum / SW, valid if iwNum >= 0, iwNum % SW == 0, iw < IW.
                                    int owLo, owHi;
                                    if (SW <= 0) {
                                        owLo = 0;
                                        owHi = 0;
                                    } else {
                                        // ow >= ow_off (so iwNum >= 0)
                                        owLo = ow_off > 0 ? ow_off : 0;
                                        // ow such that iw < IW, i.e. (ow - ow_off) < IW*SW, ow < IW*SW + ow_off
                                        const int hardMax = IW * SW + ow_off;
                                        owHi = hardMax > OW ? OW : hardMax;
                                        if (owLo > owHi) owLo = owHi;
                                    }

                                    if (SW == 1) {
                                        // iw = ow - ow_off, contiguous
                                        const float* __restrict__ src = in_row - ow_off;
                                        for (int ow = owLo; ow < owHi; ++ow) {
                                            rowAcc[ow] += src[ow] * w;
                                        }
                                    } else {
                                        for (int ow = owLo; ow < owHi; ++ow) {
                                            const int iwNum = ow - ow_off;
                                            if (iwNum < 0 || (iwNum % SW) != 0) continue;
                                            const int iw = iwNum / SW;
                                            if (iw < 0 || iw >= IW) continue;
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
