//
//  Conv3DBufExecution.cpp
//  MNN
//
//  Native 3D convolution for the OpenCL buffer backend.
//
//  Layout: NC4DHW4 packed (matches MNN's standard OpenCL buffer
//  layout for 5D tensors).  Weights packed at session-init time
//  into [Cout_block, Kd, Kh, Kw, Cin_block, 4_cin, 4_cout] so the
//  inner loop is 4 vload4 + 4 mad per (kd, kh, kw, ic_blk) tile.

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/Conv3DBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

static inline int up_div(int x, int d) {
    return (x + d - 1) / d;
}

static inline uint16_t float_to_half(float f) {
    /* IEEE 754 round-to-nearest cast for finite values; flushes
     * sub-normals to zero (Conv3D weights are bounded). */
    const uint32_t b   = *reinterpret_cast<const uint32_t*>(&f);
    const uint32_t sgn = (b >> 16) & 0x8000;
    const int32_t  e   = ((b >> 23) & 0xff) - 127 + 15;
    const uint32_t m   = b & 0x7fffff;

    if (e <= 0) {
        return static_cast<uint16_t>(sgn);
    }

    if (e >= 31) {
        return static_cast<uint16_t>(sgn | 0x7c00);
    }

    return static_cast<uint16_t>(sgn | (e << 10) | (m >> 13));
}

Conv3DBufExecution::Conv3DBufExecution(const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs,
                                       const MNN::Op* op,
                                       Backend* backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend*>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    mUnits.resize(1);
    auto& unit = mUnits[0];

    const auto* conv3d = op->main_as_Convolution3D();
    MNN_ASSERT(conv3d != nullptr);
    const auto* common = conv3d->common();

    mKernelD = common->kernels()->Get(0);
    mKernelH = common->kernels()->Get(1);
    mKernelW = common->kernels()->Get(2);
    mStrideD = common->strides()->Get(0);
    mStrideH = common->strides()->Get(1);
    mStrideW = common->strides()->Get(2);
    mPadD    = common->pads()->Get(0);
    mPadH    = common->pads()->Get(1);
    mPadW    = common->pads()->Get(2);

    mOutputChannels = common->outputCount();
    mInputChannels  = common->inputCount();

    if (mInputChannels == 0 && !inputs.empty()) {
        mInputChannels = inputs[0]->length(1);
    }

    if (mOutputChannels == 0 && !outputs.empty()) {
        mOutputChannels = outputs[0]->length(1);
    }

    const int Cin       = mInputChannels;
    const int Cout      = mOutputChannels;
    const int Cin_blk   = up_div(Cin, 4);
    const int Cout_blk  = up_div(Cout, 4);
    const int Kd        = mKernelD;
    const int Kh        = mKernelH;
    const int Kw        = mKernelW;

    const float* w_host = conv3d->weight()->data();
    const float* b_host = conv3d->bias() ? conv3d->bias()->data() : nullptr;

    /* ----- Pack weights from [Cout, Cin, Kd, Kh, Kw] -> ---------------
     *       [Cout_blk, Kd, Kh, Kw, Cin_blk, 4_cin, 4_cout]
     * Padding rows where the original Cin/Cout aren't multiples of 4
     * are zero-initialised so the kernel can ignore boundaries. */
    const size_t packed_count = static_cast<size_t>(Cout_blk) * 4 * Kd * Kh * Kw * Cin_blk * 4;
    const size_t fbytes       = (mOpenCLBackend->getPrecision() == BackendConfig::Precision_Low)
                                ? sizeof(uint16_t)
                                : sizeof(float);

    /* Allocate device-side weight buffer (no CL_MEM_ALLOC_HOST_PTR;
     * NVIDIA's OpenCL appears to mishandle host-mapped CL_MEM_READ_ONLY
     * buffers for kernels with non-trivial inner-loop access patterns
     * -- the working ConvBufExecution.cpp pattern is to allocate via
     * the MNN buffer pool with STATIC lifetime, then use a TEMP host
     * mapped buffer for the host->device copy. We follow the same
     * pattern: build the packed layout in a host-side vector, then
     * enqueueWriteBuffer it into a plain device buffer.) */
    std::vector<uint8_t> host_packed(packed_count * fbytes, 0);

    {
        const int Kvol = Kd * Kh * Kw;

        for (int oc = 0; oc < Cout; ++oc) {
            for (int ic = 0; ic < Cin; ++ic) {
                for (int kdi = 0; kdi < Kd; ++kdi) {
                    for (int khi = 0; khi < Kh; ++khi) {
                        for (int kwi = 0; kwi < Kw; ++kwi) {
                            const int src_idx =
                                ((oc * Cin + ic) * Kd + kdi) * Kh * Kw +
                                khi * Kw + kwi;

                            const int oc_blk    = oc / 4;
                            const int oc_inner  = oc % 4;
                            const int ic_blk    = ic / 4;
                            const int ic_inner  = ic % 4;

                            const int dst_idx =
                                ((((oc_blk * Kvol)
                                   + (kdi * Kh + khi) * Kw + kwi) * Cin_blk
                                  + ic_blk) * 16)
                                + ic_inner * 4 + oc_inner;

                            const float v = w_host[src_idx];

                            if (mOpenCLBackend->getPrecision() == BackendConfig::Precision_Low) {
                                reinterpret_cast<uint16_t*>(host_packed.data())[dst_idx] = float_to_half(v);
                            } else {
                                reinterpret_cast<float*>(host_packed.data())[dst_idx] = v;
                            }
                        }
                    }
                }
            }
        }
    }

    mKernelBuffer.reset(new cl::Buffer(
        runtime->context(), CL_MEM_READ_WRITE,
        packed_count * fbytes));
    {
        cl_int err = runtime->commandQueue().enqueueWriteBuffer(
            *mKernelBuffer, CL_TRUE, 0, packed_count * fbytes, host_packed.data());
        MNN_CHECK_CL_SUCCESS(err, "write Conv3D weight buffer");
    }

    /* ----- Bias buffer (same device-only-no-ALLOC_HOST_PTR pattern). */
    const size_t bias_count = static_cast<size_t>(Cout_blk) * 4;
    std::vector<uint8_t> host_bias(bias_count * fbytes, 0);

    if (mOpenCLBackend->getPrecision() == BackendConfig::Precision_Low) {
        auto* dst = reinterpret_cast<uint16_t*>(host_bias.data());
        for (int c = 0; c < Cout; ++c) {
            dst[c] = float_to_half(b_host ? b_host[c] : 0.0f);
        }
    } else {
        auto* dst = reinterpret_cast<float*>(host_bias.data());
        for (int c = 0; c < Cout; ++c) {
            dst[c] = b_host ? b_host[c] : 0.0f;
        }
    }

    mBiasBuffer.reset(new cl::Buffer(
        runtime->context(), CL_MEM_READ_WRITE,
        bias_count * fbytes));
    {
        cl_int err = runtime->commandQueue().enqueueWriteBuffer(
            *mBiasBuffer, CL_TRUE, 0, bias_count * fbytes, host_bias.data());
        MNN_CHECK_CL_SUCCESS(err, "write Conv3D bias buffer");
    }

    /* Build BOTH kernel variants at ctor; per-encode we pick which to
     * launch based on whether the input layout is NC4HW4 (default) or
     * NCHW (when reading directly from a MEMORY_VIRTUAL Region's
     * NCHW-formatted source tensor). */
    std::set<std::string> buildOptions;
    mKernelNC4 = runtime->buildKernel("conv_3d_buf", "conv_3d_buf_nc4dhw4",
                                      buildOptions, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(mKernelNC4);
    mMaxWGS_NC4 = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelNC4));

    mKernelNCHW = runtime->buildKernel("conv_3d_buf", "conv_3d_buf_nchw_in",
                                       buildOptions, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(mKernelNCHW);
    mMaxWGS_NCHW = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelNCHW));

    /* unit.kernel will be assigned per-encode. */
    unit.kernel = mKernelNC4;  /* default; onEncode may swap */
    mMaxWorkGroupSize = mMaxWGS_NC4;
}

ErrorCode Conv3DBufExecution::onEncode(const std::vector<Tensor*>& inputs,
                                       const std::vector<Tensor*>& outputs) {
    auto& unit  = mUnits[0];
    Tensor* in  = inputs[0];
    Tensor* out = outputs[0];
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    const int Cin     = in->length(1);
    const int Din     = in->length(2);
    const int Hin     = in->length(3);
    const int Win     = in->length(4);
    const int Cout    = out->length(1);
    const int Dout    = out->length(2);
    const int Hout    = out->length(3);
    const int Wout    = out->length(4);
    const int Cinblk  = up_div(Cin, 4);
    const int Coutblk = up_div(Cout, 4);

    mGWS[0] = static_cast<uint32_t>(Wout);
    mGWS[1] = static_cast<uint32_t>(Dout * Hout);
    mGWS[2] = static_cast<uint32_t>(Coutblk);

    /* Choose between the NC4HW4-reader kernel and the NCHW-reader
     * kernel based on input layout. Per-encode decision because each
     * Conv3D op may have a different upstream producer. Currently only
     * the NCHW path reads origin directly (saves one Raster); the
     * NC4HW4 path reads MNN's materialized buffer like the original. */
    auto in_d = TensorUtils::getDescribe(in);
    Tensor* read_from = in;
    bool use_nchw_kernel = false;
    if (in_d->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL
        && !in_d->regions.empty()) {
        const auto& r = in_d->regions[0];
        bool identity = (r.src.offset == 0 && r.dst.offset == 0
                         && r.src.stride[0] == r.dst.stride[0]
                         && r.src.stride[1] == r.dst.stride[1]
                         && r.src.stride[2] == r.dst.stride[2]);
        if (identity && r.origin && r.origin->deviceId() != 0) {
            auto orig_d = TensorUtils::getDescribe(r.origin);
            if (orig_d->dimensionFormat != MNN_DATA_FORMAT_NC4HW4) {
                read_from = r.origin;
                use_nchw_kernel = true;
            }
        }
    }

    if (use_nchw_kernel) {
        unit.kernel = mKernelNCHW;
        mMaxWorkGroupSize = mMaxWGS_NCHW;
    } else {
        unit.kernel = mKernelNC4;
        mMaxWorkGroupSize = mMaxWGS_NC4;
    }

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGWS[0]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[1]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(read_from));
    ret |= unit.kernel->get().setArg(idx++, *mKernelBuffer);
    ret |= unit.kernel->get().setArg(idx++, *mBiasBuffer);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(out));
    /* NC4HW4 kernel expects in_cblock; NCHW kernel expects raw Cin. */
    ret |= unit.kernel->get().setArg(idx++, use_nchw_kernel
                                            ? static_cast<int>(Cin)
                                            : static_cast<int>(Cinblk));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Din));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Hin));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Win));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Dout));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Hout));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Wout));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mKernelD));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mKernelH));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mKernelW));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mStrideD));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mStrideH));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mStrideW));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mPadD));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mPadH));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(mPadW));
    MNN_CHECK_CL_SUCCESS(ret, "setArg Conv3DBufExecution");

    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime,
                            use_nchw_kernel ? "conv_3d_buf_nchw_in"
                                            : "conv_3d_buf_nc4dhw4",
                            unit.kernel,
                            mOpenCLBackend->getCLTuneLevel(),
                            "conv_3d_buf").first;

    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize  = {mLWS[0], mLWS[1], mLWS[2]};
    return NO_ERROR;
}

class Conv3DBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~Conv3DBufCreator() = default;
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op,
                                Backend* backend) const override {
        OPENCL_CREATOR_CHECK(new Conv3DBufExecution(inputs, outputs, op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(Conv3DBufCreator, OpType_Convolution3D, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
