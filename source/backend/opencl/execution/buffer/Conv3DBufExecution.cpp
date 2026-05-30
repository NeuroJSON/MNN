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

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <map>
#include <tuple>
#include <vector>

namespace MNN {
namespace OpenCL {

/* Shape signature used by the optional MNN_CONV3D_TIMING aggregator
 * (declared at file scope so the std::sort lambda below is portable
 * across the C++11 / C++14 dialects MNN's own build picks per file). */
typedef std::tuple<int, int, int, int, int, int, int, int> Conv3DSig;

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

    mKernelNC4T22 = runtime->buildKernel("conv_3d_buf", "conv_3d_buf_nc4dhw4_t22",
                                         buildOptions, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(mKernelNC4T22);
    mMaxWGS_NC4T22 = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(mKernelNC4T22));

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

    /* Default GWS for the single-output kernel. The tiled kernel below
     * halves dim0/dim1 if selected. */
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
    /* Read-from-origin shortcut is only safe for SINGLE-region
     * MEMORY_VIRTUAL inputs. Multi-region inputs (e.g. from Concat,
     * Slice, BinaryOp's broadcast-rasters, etc.) require the Raster
     * materialization to actually run -- our kernel only consumes
     * regions[0]->origin and would read garbage for channel ranges
     * supplied by regions[1+]. Empirical SIAM test: at S64-scale
     * mini-U-Net, the post-Concat Conv3D faults with CL_OUT_OF_RESOURCES
     * when this guard is absent. */
    if (in_d->memoryType == Tensor::InsideDescribe::MEMORY_VIRTUAL
        && in_d->regions.size() == 1) {
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

    /* When the input is NC4HW4 (the common case) and the output has
     * at least 2 spatial cells along W (so the 2x2 tile is worth
     * launching), use the tiled kernel. Disable via MNN_CONV3D_NOTILE=1
     * for A/B comparison or to bypass if a regression is found. */
    static int s_notile = -1;
    if (s_notile < 0) {
        const char* e = std::getenv("MNN_CONV3D_NOTILE");
        s_notile = (e && e[0] == '1') ? 1 : 0;
    }
    const bool can_tile = !use_nchw_kernel && (Wout >= 2) && (s_notile == 0);

    if (use_nchw_kernel) {
        unit.kernel = mKernelNCHW;
        mMaxWorkGroupSize = mMaxWGS_NCHW;
    } else if (can_tile) {
        unit.kernel = mKernelNC4T22;
        mMaxWorkGroupSize = mMaxWGS_NC4T22;
        /* Halve dim0 (ceil) and dim1's Hout factor (ceil). */
        mGWS[0] = static_cast<uint32_t>((Wout + 1) >> 1);
        mGWS[1] = static_cast<uint32_t>(Dout * ((Hout + 1) >> 1));
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

    const char* kname = use_nchw_kernel ? "conv_3d_buf_nchw_in"
                        : (can_tile ? "conv_3d_buf_nc4dhw4_t22"
                                    : "conv_3d_buf_nc4dhw4");
    mLWS = localWS3DDefault(mGWS, mMaxWorkGroupSize, runtime,
                            kname,
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

ErrorCode Conv3DBufExecution::onExecute(const std::vector<Tensor*>& inputs,
                                        const std::vector<Tensor*>& outputs) {
    /* Optional per-op output sampling for debugging. Set MNN_CONV3D_PROBE=1
     * in env to print the first 8 output floats after each Conv3D fires;
     * useful for narrowing down which op in a pipeline first produces
     * zero / NaN output. */
    static int s_probe = -1;
    if (s_probe < 0) {
        const char* e = std::getenv("MNN_CONV3D_PROBE");
        s_probe = (e && e[0] == '1') ? 1 : 0;
    }
    /* Optional per-op timing via clFinish bracketing. Set MNN_CONV3D_TIMING=1.
     * Accumulates wall-clock per (Cin, Cout, Dout, Hout, Wout, stride) shape
     * signature and prints a summary at program exit (atexit) — useful for
     * identifying the top-cost Conv3D shapes when optimizing kernels. */
    static int s_timing = -1;
    if (s_timing < 0) {
        const char* e = std::getenv("MNN_CONV3D_TIMING");
        s_timing = (e && e[0] == '1') ? 1 : 0;
    }

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto q = runtime->commandQueue();
    /* Accumulator keyed by Conv3DSig (Cin,Cout,Dout,Hout,Wout,sd,sh,sw).
     * Defined at file scope so a portable comparator type can be named. */
    static std::map<Conv3DSig, std::pair<double, int>> s_acc; // total_us, n_calls
    double t0 = 0.0;
    if (s_timing) {
        q.finish();
        t0 = static_cast<double>(clock());
    }

    auto err = CommonExecution::onExecute(inputs, outputs);

    if (s_timing) {
        q.finish();
        double dt_us = (static_cast<double>(clock()) - t0) / CLOCKS_PER_SEC * 1.0e6;
        Conv3DSig sig{mInputChannels, mOutputChannels,
                      outputs[0]->length(2), outputs[0]->length(3), outputs[0]->length(4),
                      mStrideD, mStrideH, mStrideW};
        auto& slot = s_acc[sig];
        slot.first  += dt_us;
        slot.second += 1;
        static bool s_registered = false;
        if (!s_registered) {
            s_registered = true;
            std::atexit([]() {
                fprintf(stderr, "\n[Conv3D::timing] shape (Cin,Cout,Dout,Hout,Wout,sd,sh,sw)  n_calls  total_ms  avg_us\n");
                std::vector<std::pair<Conv3DSig, std::pair<double, int>>> items(s_acc.begin(), s_acc.end());
                std::sort(items.begin(), items.end(),
                          [](const std::pair<Conv3DSig, std::pair<double, int>>& a,
                             const std::pair<Conv3DSig, std::pair<double, int>>& b) {
                              return a.second.first > b.second.first;
                          });
                double grand = 0;
                for (auto& it : items) grand += it.second.first;
                for (auto& it : items) {
                    auto& s = it.first;
                    fprintf(stderr, "  Cin=%-4d Cout=%-4d D=%-3d H=%-3d W=%-3d s=(%d,%d,%d)  n=%-4d total=%7.1f ms avg=%6.1f us  pct=%5.1f%%\n",
                            std::get<0>(s), std::get<1>(s), std::get<2>(s), std::get<3>(s), std::get<4>(s),
                            std::get<5>(s), std::get<6>(s), std::get<7>(s),
                            it.second.second, it.second.first / 1000.0,
                            it.second.first / it.second.second,
                            100.0 * it.second.first / grand);
                }
                fprintf(stderr, "  [Conv3D total]: %.1f ms across %zu shapes\n",
                        grand / 1000.0, s_acc.size());
            });
        }
    }

    if (s_probe) {
        auto& out_buf = openCLBuffer(outputs[0]);
        float buf[8] = {0};
        cl_int cret = q.enqueueReadBuffer(out_buf, CL_TRUE, 0, sizeof(buf), buf);
        int nz = 0;
        for (int i = 0; i < 8; ++i) if (buf[i] != 0.0f) nz++;
        fprintf(stderr,
                "[Conv3D::probe] Cin=%d Cout=%d s=(%d,%d,%d) out=(D=%d,H=%d,W=%d) "
                "read_ret=%d nz=%d/8 first4=(%g,%g,%g,%g)\n",
                mInputChannels, mOutputChannels,
                mStrideD, mStrideH, mStrideW,
                outputs[0]->length(2), outputs[0]->length(3), outputs[0]->length(4),
                (int)cret, nz, buf[0], buf[1], buf[2], buf[3]);
    }
    return err;
}

REGISTER_OPENCL_OP_CREATOR(Conv3DBufCreator, OpType_Convolution3D, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
