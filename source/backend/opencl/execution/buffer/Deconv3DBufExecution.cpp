//
//  Deconv3DBufExecution.cpp
//  MNN
//
//  Native 3D transposed convolution for the OpenCL buffer backend.
//  See Deconv3DBufExecution.hpp for the why & how.

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/Deconv3DBufExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

static inline int up_div(int x, int d) {
    return (x + d - 1) / d;
}

static inline uint16_t float_to_half(float f) {
    /* Same IEEE 754 round-to-nearest cast as Conv3DBufExecution; flushes
     * sub-normals to zero (ConvTranspose weights are bounded). */
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

Deconv3DBufExecution::Deconv3DBufExecution(const std::vector<Tensor*>& inputs,
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

    /* Pack weights from ONNX (Cin, Cout, Kd, Kh, Kw) into the same
     * kernel-friendly layout Conv3DBufExecution uses:
     *   [Cout_blk, Kd, Kh, Kw, Cin_blk, 4_cin, 4_cout]
     * The kernel reads this exactly the same way as the Conv3D kernel
     * -- the only difference is the SOURCE addressing of the unpacked
     * weight (ONNX Conv3D = (Cout, Cin, ...); ConvTranspose3D = (Cin,
     * Cout, ...)). Tail rows where original Cin/Cout aren't multiples
     * of 4 stay zero so the kernel can ignore boundary handling. */
    const size_t packed_count = static_cast<size_t>(Cout_blk) * 4 * Kd * Kh * Kw * Cin_blk * 4;
    const size_t fbytes       = (mOpenCLBackend->getPrecision() == BackendConfig::Precision_Low)
                                ? sizeof(uint16_t)
                                : sizeof(float);

    std::vector<uint8_t> host_packed(packed_count * fbytes, 0);

    {
        const int Kvol = Kd * Kh * Kw;

        for (int ic = 0; ic < Cin; ++ic) {
            for (int oc = 0; oc < Cout; ++oc) {
                for (int kdi = 0; kdi < Kd; ++kdi) {
                    for (int khi = 0; khi < Kh; ++khi) {
                        for (int kwi = 0; kwi < Kw; ++kwi) {
                            /* ONNX ConvTranspose3D weight layout:
                             *   (Cin, Cout, Kd, Kh, Kw) */
                            const int src_idx =
                                ((ic * Cout + oc) * Kd + kdi) * Kh * Kw +
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
        MNN_CHECK_CL_SUCCESS(err, "write Deconv3D weight buffer");
    }

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
        MNN_CHECK_CL_SUCCESS(err, "write Deconv3D bias buffer");
    }

    std::set<std::string> buildOptions;
    mKernelNC4 = runtime->buildKernel("deconv_3d_buf", "deconv_3d_buf_nc4dhw4",
                                      buildOptions, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(mKernelNC4);
    mMaxWGS_NC4 = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelNC4));

    mKernelNCHW = runtime->buildKernel("deconv_3d_buf", "deconv_3d_buf_nchw",
                                       buildOptions, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(mKernelNCHW);
    mMaxWGS_NCHW = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelNCHW));

    unit.kernel = mKernelNC4;
    mMaxWorkGroupSize = mMaxWGS_NC4;
}

ErrorCode Deconv3DBufExecution::onEncode(const std::vector<Tensor*>& inputs,
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

    /* Select kernel variant by input layout:
     *   NC4HW4-tagged inputs use the packed-read / packed-write kernel.
     *   NCHW-tagged inputs (typical for SIAM where a ConvertTensor
     *   materializes the input into NCHW before ConvTranspose3D) use
     *   the strided-read / strided-write kernel.
     * Choice is per-op because each ConvTranspose3D's upstream
     * producer can differ across the network. */
    const auto in_fmt = TensorUtils::getDescribe(in)->dimensionFormat;
    const bool use_nchw_kernel = (in_fmt != MNN_DATA_FORMAT_NC4HW4);

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
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(in));
    ret |= unit.kernel->get().setArg(idx++, *mKernelBuffer);
    ret |= unit.kernel->get().setArg(idx++, *mBiasBuffer);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(out));
    /* NC4HW4 kernel takes in_cblock; NCHW kernel takes raw Cin. */
    ret |= unit.kernel->get().setArg(idx++, use_nchw_kernel
                                            ? static_cast<int>(Cin)
                                            : static_cast<int>(Cinblk));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Din));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Hin));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Win));
    /* NCHW kernel needs out_c too (for tail boundary on Cout%4); the
     * NC4HW4 kernel ignores it (always writes full float4). */
    if (use_nchw_kernel) {
        ret |= unit.kernel->get().setArg(idx++, static_cast<int>(Cout));
    }
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
    MNN_CHECK_CL_SUCCESS(ret, "setArg Deconv3DBufExecution");

    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime,
                            use_nchw_kernel ? "deconv_3d_buf_nchw"
                                            : "deconv_3d_buf_nc4dhw4",
                            unit.kernel,
                            mOpenCLBackend->getCLTuneLevel(),
                            "deconv_3d_buf").first;

    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize  = {mLWS[0], mLWS[1], mLWS[2]};
    return NO_ERROR;
}

ErrorCode Deconv3DBufExecution::onExecute(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs) {
    /* Mirrors the MNN_CONV3D_PROBE pattern: set MNN_DECONV3D_PROBE=1
     * to print the first 4 output values per op for narrowing
     * down which decoder ConvTranspose first goes silent. */
    static int s_probe = -1;
    if (s_probe < 0) {
        const char* e = std::getenv("MNN_DECONV3D_PROBE");
        s_probe = (e && e[0] == '1') ? 1 : 0;
    }

    auto err = CommonExecution::onExecute(inputs, outputs);
    if (s_probe) {
        auto& out_buf = openCLBuffer(outputs[0]);
        float buf[8] = {0};
        auto q = mOpenCLBackend->getOpenCLRuntime()->commandQueue();
        cl_int cret = q.enqueueReadBuffer(out_buf, CL_TRUE, 0, sizeof(buf), buf);
        int nz = 0;
        for (int i = 0; i < 8; ++i) if (buf[i] != 0.0f) nz++;
        fprintf(stderr,
                "[Deconv3D::probe] Cin=%d Cout=%d s=(%d,%d,%d) out=(D=%d,H=%d,W=%d) "
                "read_ret=%d nz=%d/8 first4=(%g,%g,%g,%g)\n",
                mInputChannels, mOutputChannels,
                mStrideD, mStrideH, mStrideW,
                outputs[0]->length(2), outputs[0]->length(3), outputs[0]->length(4),
                (int)cret, nz, buf[0], buf[1], buf[2], buf[3]);
    }
    return err;
}

class Deconv3DBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~Deconv3DBufCreator() = default;
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                const MNN::Op* op,
                                Backend* backend) const override {
        OPENCL_CREATOR_CHECK(new Deconv3DBufExecution(inputs, outputs, op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(Deconv3DBufCreator, OpType_ConvTranspose3D, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
