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

    // Outer-product blocking: process OC_BLOCK output channels per inner
    // iteration. Each input load feeds OC_BLOCK FMAs (memory bandwidth
    // amortization). SIAM's OC values (32, 64, 128, 256, 320) are all
    // multiples of 4 -- empirically OC_BLOCK=4 outperforms 8 on AVX2
    // x86_64: OC=8 doubles the accumulator buffer (still fits in L1 but
    // straddles more cache lines) and exhausts AVX2's 16 ymm registers
    // once you include input broadcast + weight broadcasts, causing
    // register spills. The partial-block path is kept for other models.
    constexpr int OC_BLOCK = 4;
    const int ocBlocks = (OC + OC_BLOCK - 1) / OC_BLOCK;

    for (int n = 0; n < N; ++n) {
        const float* __restrict__ in_n  = inputData + n * inStrideN;
        float*       __restrict__ out_n = outputData + n * outStrideN;

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            std::vector<float> rowAccBuf(static_cast<size_t>(OC_BLOCK) * OW, 0.0f);
            float* const acc0 = rowAccBuf.data() + 0 * OW;
            float* const acc1 = rowAccBuf.data() + 1 * OW;
            float* const acc2 = rowAccBuf.data() + 2 * OW;
            float* const acc3 = rowAccBuf.data() + 3 * OW;

            const int totalUnits = ocBlocks * OD;
            for (int unit = static_cast<int>(tId); unit < totalUnits; unit += threadNumber) {
                const int ocb   = unit / OD;
                const int od    = unit % OD;
                const int oc_lo = ocb * OC_BLOCK;
                const int oc_hi = std::min(oc_lo + OC_BLOCK, OC);
                const int oc_n  = oc_hi - oc_lo;

                int kdLo = 0;
                int kdHi = KD;
                {
                    const int id_at_0 = od * SD - padFrontD;
                    while (kdLo < KD && id_at_0 + kdLo * DD < 0) ++kdLo;
                    while (kdHi > kdLo && id_at_0 + (kdHi - 1) * DD >= ID) --kdHi;
                }

                const float* __restrict__ w_oc0 = weightData + (oc_lo + 0) * wStrideOC;
                const float* __restrict__ w_oc1 = oc_n > 1 ? weightData + (oc_lo + 1) * wStrideOC : w_oc0;
                const float* __restrict__ w_oc2 = oc_n > 2 ? weightData + (oc_lo + 2) * wStrideOC : w_oc0;
                const float* __restrict__ w_oc3 = oc_n > 3 ? weightData + (oc_lo + 3) * wStrideOC : w_oc0;

                const float bias0 = biasData[oc_lo + 0];
                const float bias1 = oc_n > 1 ? biasData[oc_lo + 1] : 0.0f;
                const float bias2 = oc_n > 2 ? biasData[oc_lo + 2] : 0.0f;
                const float bias3 = oc_n > 3 ? biasData[oc_lo + 3] : 0.0f;

                for (int oh = 0; oh < OH; ++oh) {
                    int khLo = 0;
                    int khHi = KH;
                    {
                        const int ih_at_0 = oh * SH - padTopH;
                        while (khLo < KH && ih_at_0 + khLo * DH < 0) ++khLo;
                        while (khHi > khLo && ih_at_0 + (khHi - 1) * DH >= IH) --khHi;
                    }

                    for (int ow = 0; ow < OW; ++ow) acc0[ow] = bias0;
                    for (int ow = 0; ow < OW; ++ow) acc1[ow] = bias1;
                    for (int ow = 0; ow < OW; ++ow) acc2[ow] = bias2;
                    for (int ow = 0; ow < OW; ++ow) acc3[ow] = bias3;

                    for (int ic = 0; ic < IC; ++ic) {
                        const float* __restrict__ in_ic_n = in_n + ic * inStrideC;
                        const std::int64_t w_ic_off = static_cast<std::int64_t>(ic) * wStrideIC;

                        for (int kd = kdLo; kd < kdHi; ++kd) {
                            const int id = od * SD - padFrontD + kd * DD;
                            const float* __restrict__ in_d = in_ic_n
                                                             + static_cast<std::int64_t>(id) * IH * IW;
                            const std::int64_t w_kd_off = w_ic_off + static_cast<std::int64_t>(kd) * wStrideKD;

                            for (int kh = khLo; kh < khHi; ++kh) {
                                const int ih = oh * SH - padTopH + kh * DH;
                                const float* __restrict__ in_row = in_d
                                                                   + static_cast<std::int64_t>(ih) * IW;
                                const std::int64_t w_kh_off = w_kd_off + static_cast<std::int64_t>(kh) * KW;

                                for (int kw = 0; kw < KW; ++kw) {
                                    const float w0 = w_oc0[w_kh_off + kw];
                                    const float w1 = w_oc1[w_kh_off + kw];
                                    const float w2 = w_oc2[w_kh_off + kw];
                                    const float w3 = w_oc3[w_kh_off + kw];

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
                                        const float* __restrict__ src = in_row + iw_off;
                                        for (int ow = owLo; ow < owHi; ++ow) {
                                            const float v = src[ow];
                                            acc0[ow] += v * w0;
                                            acc1[ow] += v * w1;
                                            acc2[ow] += v * w2;
                                            acc3[ow] += v * w3;
                                        }
                                    } else {
                                        for (int ow = owLo; ow < owHi; ++ow) {
                                            const int iw = ow * SW + iw_off;
                                            const float v = in_row[iw];
                                            acc0[ow] += v * w0;
                                            acc1[ow] += v * w1;
                                            acc2[ow] += v * w2;
                                            acc3[ow] += v * w3;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Write back OC_BLOCK rows -- one per channel in the block.
                    const float* const accs[OC_BLOCK] = {acc0, acc1, acc2, acc3};
                    for (int ocb_in = 0; ocb_in < oc_n; ++ocb_in) {
                        const int oc = oc_lo + ocb_in;
                        const float* __restrict__ src = accs[ocb_in];
                        float* __restrict__ out_row = out_n + oc * outStrideC
                                                     + (static_cast<std::int64_t>(od) * OH + oh) * OW;
                        if (relu6) {
                            for (int ow = 0; ow < OW; ++ow) {
                                float v = src[ow];
                                if (v < 0.0f) v = 0.0f;
                                else if (v > 6.0f) v = 6.0f;
                                out_row[ow] = v;
                            }
                        } else if (relu) {
                            for (int ow = 0; ow < OW; ++ow) {
                                float v = src[ow];
                                out_row[ow] = v < 0.0f ? 0.0f : v;
                            }
                        } else {
                            for (int ow = 0; ow < OW; ++ow) {
                                out_row[ow] = src[ow];
                            }
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
