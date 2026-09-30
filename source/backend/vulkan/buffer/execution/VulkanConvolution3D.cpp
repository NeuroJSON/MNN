//
//  VulkanConvolution3D.cpp
//  MNN
//
//  Native Convolution3D / ConvTranspose3D for the Vulkan BUFFER backend.
//  Vulkan counterpart of opencl/execution/buffer/{Conv3D,Deconv3D}BufExecution.
//
//  Without it the Vulkan backend has no creator for either op: with the
//  geometry decomposition disabled (SIAM_DISABLE_GEOM_CONV3D /
//  SIAM_DISABLE_GEOM_DECONV3D, which siamize sets) every Conv3D falls back to
//  the CPU with a device<->host round trip per op; with it enabled the op
//  expands into ~40 Conv2D/Im2Col/Raster launches plus multi-GB workspaces.
//
//  The shader is glsl/conv3d.comp. Kernel / stride / pad / layouts / conv-vs-
//  transposed / tiling are specialization constants, so each distinct layer
//  shape gets its own fully-unrolled pipeline (cached by the pipeline factory).
//
//  Supported: group = 1, dilation = 1, explicit pads (PadMode_CAFFE), fp32 or
//  fp16 storage (fp32 accumulation either way), NC4HW4 or NCHW input/output.
//  Anything else returns nullptr from the creator so MNN falls back to CPU.
//

#include "VulkanBasicExecution.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace MNN {

namespace {

struct Conv3DParam {
    ivec4 inSize;   // W, H, D, C4
    ivec4 outSize;  // W, H, D, C4
    ivec4 chan;     // Cin, Cout, batch, OC tiles
    ivec4 work;     // W tiles, total work-items, 0, 0
};

int envInt(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? std::atoi(v) : fallback;
}

} // namespace

class VulkanConvolution3D : public VulkanBasicExecution {
public:
    VulkanConvolution3D(Backend* bn, const Convolution3DCommon* common, bool transposed, int ci, int co)
        : VulkanBasicExecution(bn), mTransposed(transposed), mCin(ci), mCout(co) {
        for (int i = 0; i < 3; ++i) {
            mKernel[i] = common->kernels()->Get(i);
            mStride[i] = common->strides()->Get(i);
            // pads is (D, H, W) or (D, H, W) begin + (D, H, W) end; only the
            // begin pads enter the index math, the output extent is taken
            // from the output tensor.
            mPad[i]    = common->pads()->Get(i);
        }
        mActivation = common->relu6() ? 2 : (common->relu() ? 1 : 0);
        mWTile  = envInt("MNN_VK_CONV3D_WTILE", 2);
        mOCTile = envInt("MNN_VK_CONV3D_OCTILE", 4);
        mLocal  = envInt("MNN_VK_CONV3D_LOCAL", 64);
        if (mWTile < 1)  mWTile = 1;
        if (mOCTile < 1) mOCTile = 1;
        if (mLocal < 1)  mLocal = 64;
        mUniform = static_cast<VulkanBackend*>(bn)->allocUniform();
    }

    ~VulkanConvolution3D() override {
        static_cast<VulkanBackend*>(backend())->recycleUniform(mUniform);
    }

    // Repack the flatbuffer weights into the shader's layout and upload them
    // (plus bias) to device-local memory. Conv3D weights are (Cout, Cin, K);
    // ConvTranspose3D weights are (Cin, Cout, K).
    bool uploadWeights(const float* weight, const float* bias) {
        auto vkBn = static_cast<VulkanBackend*>(backend());
        const bool fp16 = vkBn->useFP16();
        const int kvol  = mKernel[0] * mKernel[1] * mKernel[2];
        const int ci4   = UP_DIV(mCin, 4);
        const int co4   = UP_DIV(mCout, 4);
        const size_t count = (size_t)co4 * kvol * ci4 * 16;
        std::vector<float> packed(count, 0.0f);
        for (int oc = 0; oc < mCout; ++oc) {
            for (int ic = 0; ic < mCin; ++ic) {
                const float* src = mTransposed ? weight + ((size_t)ic * mCout + oc) * kvol
                                               : weight + ((size_t)oc * mCin + ic) * kvol;
                for (int k = 0; k < kvol; ++k) {
                    const size_t dst = (((size_t)(oc / 4) * kvol + k) * ci4 + ic / 4) * 16 + (ic % 4) * 4 + (oc % 4);
                    packed[dst] = src[k];
                }
            }
        }
        std::vector<float> biasPacked((size_t)co4 * 4, 0.0f);
        if (nullptr != bias) {
            ::memcpy(biasPacked.data(), bias, mCout * sizeof(float));
        }
        mWeightBuffer = upload(packed, fp16);
        mBiasBuffer   = upload(biasPacked, fp16);
        return nullptr != mWeightBuffer && nullptr != mBiasBuffer;
    }

    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override {
        auto vkBn   = static_cast<VulkanBackend*>(backend());
        auto input  = inputs[0];
        auto output = outputs[0];
        if (input->dimensions() != 5 || output->dimensions() != 5) {
            return NOT_SUPPORT;
        }
        const int batch = input->length(0);
        const int inC4  = UP_DIV(input->length(1), 4);
        const int outC4 = UP_DIV(output->length(1), 4);
        const bool inNC4  = TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
        const bool outNC4 = TensorUtils::getDescribe(output)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;

        Conv3DParam param;
        ::memset(&param, 0, sizeof(param));
        param.inSize[0]  = input->length(4);
        param.inSize[1]  = input->length(3);
        param.inSize[2]  = input->length(2);
        param.inSize[3]  = inC4;
        param.outSize[0] = output->length(4);
        param.outSize[1] = output->length(3);
        param.outSize[2] = output->length(2);
        param.outSize[3] = outC4;
        const int wTiles  = UP_DIV(param.outSize[0], mWTile);
        const int ocTiles = UP_DIV(outC4, mOCTile);
        const size_t total = (size_t)batch * ocTiles * param.outSize[2] * param.outSize[1] * wTiles;
        if (total > (size_t)0x7fffffff) {
            return NOT_SUPPORT;
        }
        param.chan[0] = input->length(1);
        param.chan[1] = output->length(1);
        param.chan[2] = batch;
        param.chan[3] = ocTiles;
        param.work[0] = wTiles;
        param.work[1] = (int)total;
        {
            auto ptr = mUniform->map();
            ::memcpy(ptr, &param, sizeof(param));
            mUniform->unmap();
        }

        std::vector<VkDescriptorType> types {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        };
        std::string key = vkBn->useFP16() ? "glsl_conv3d_FP16_comp" : "glsl_conv3d_comp";
        std::vector<uint32_t> spec {
            (uint32_t)mKernel[2], (uint32_t)mKernel[1], (uint32_t)mKernel[0],
            (uint32_t)mStride[2], (uint32_t)mStride[1], (uint32_t)mStride[0],
            (uint32_t)mPad[2],    (uint32_t)mPad[1],    (uint32_t)mPad[0],
            inNC4 ? 1u : 0u, outNC4 ? 1u : 0u, mTransposed ? 1u : 0u,
            (uint32_t)mWTile, (uint32_t)mOCTile, (uint32_t)mActivation
        };
        auto pipeline = vkBn->getPipeline(key, types, {(uint32_t)mLocal, 1, 1}, spec);
        if (nullptr == pipeline) {
            return NOT_SUPPORT;
        }
        mDescriptorSet.reset(pipeline->createSet());

        auto inBuf  = vkBn->getTensorBuffer(input);
        auto outBuf = vkBn->getTensorBuffer(output);
        const size_t inSize  = vkBn->getTensorSize(input);
        const size_t outSize = vkBn->getTensorSize(output);
        mDescriptorSet->writeBuffer(outBuf.first->buffer(), 0, outSize, outBuf.second);
        mDescriptorSet->writeBuffer(inBuf.first->buffer(), 1, inSize, inBuf.second);
        mDescriptorSet->writeBuffer(mWeightBuffer->buffer(), 2, mWeightBuffer->size());
        mDescriptorSet->writeBuffer(mBiasBuffer->buffer(), 3, mBiasBuffer->size());
        mDescriptorSet->writeBuffer(mUniform->buffer(), 4, mUniform->size());
        mDescriptorSet->writeBuffer(inBuf.first->buffer(), 5, inSize, inBuf.second);
        mDescriptorSet->writeBuffer(outBuf.first->buffer(), 6, outSize, outBuf.second);
        pipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

        // 1D dispatch, spilling into Y when the group count exceeds the
        // device's X limit (65535 on some Intel / mobile drivers).
        const uint32_t groups = (uint32_t)UP_DIV(total, (size_t)mLocal);
        const uint32_t maxX   = vkBn->proty().limits.maxComputeWorkGroupCount[0];
        const uint32_t gx     = groups < maxX ? groups : maxX;
        const uint32_t gy     = UP_DIV(groups, gx);
        vkCmdDispatch(cmdBuffer->get(), gx, gy, 1);
        return NO_ERROR;
    }

#ifdef ENABLE_VULKAN_TIME_PROFILE
    void setLabel(const Tensor* in, const Tensor* out) {
        auto fmt = [](const Tensor* t) {
            return TensorUtils::getDescribe(t)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 ? "nc4" : "nchw";
        };
        char label[160];
        snprintf(label, sizeof(label), "%s %d->%d k%d s%d out%dx%dx%d %s->%s", mTransposed ? "Deconv3D" : "Conv3D",
                 in->length(1), out->length(1), mKernel[0], mStride[0], out->length(2), out->length(3),
                 out->length(4), fmt(in), fmt(out));
        mLabel = label;
        setName(mLabel.c_str());
    }
#endif

private:
    std::shared_ptr<VulkanBuffer> upload(const std::vector<float>& host, bool fp16) {
        auto vkBn = static_cast<VulkanBackend*>(backend());
        const size_t bytes = host.size() * (fp16 ? sizeof(int16_t) : sizeof(float));
        // requirements_mask = 0 selects device-local memory (same as the tensor
        // allocator). The HOST_VISIBLE default would put hundreds of MB of
        // weights in system RAM on discrete GPUs.
        std::shared_ptr<VulkanBuffer> buffer(new VulkanBuffer(vkBn->getMemoryPool(), false, bytes, nullptr,
                                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                              VK_SHARING_MODE_EXCLUSIVE, 0));
        if (!buffer->valid()) {
            return nullptr;
        }
        if (fp16) {
            std::vector<int16_t> half(host.size());
            FLOAT_TO_HALF(host.data(), half.data(), (int)host.size());
            vkBn->copyToGPUBuffer(half.data(), buffer->buffer(), bytes, 0);
        } else {
            vkBn->copyToGPUBuffer(host.data(), buffer->buffer(), bytes, 0);
        }
        return buffer;
    }

    bool mTransposed;
    int mCin;
    int mCout;
    int mKernel[3];  // D, H, W
    int mStride[3];
    int mPad[3];
    int mActivation;
    int mWTile;
    int mOCTile;
    int mLocal;
    std::shared_ptr<VulkanBuffer> mWeightBuffer;
    std::shared_ptr<VulkanBuffer> mBiasBuffer;
    std::shared_ptr<VulkanBuffer> mUniform;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
#ifdef ENABLE_VULKAN_TIME_PROFILE
    std::string mLabel;
#endif
};

class VulkanConvolution3DCreator : public VulkanBackend::Creator {
public:
    explicit VulkanConvolution3DCreator(bool transposed) : mTransposed(transposed) {}

    VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op, Backend* bn) const override {
        if (std::getenv("MNN_VK_DISABLE_CONV3D") != nullptr) {
            return nullptr;
        }
        auto conv = op->main_as_Convolution3D();
        if (nullptr == conv || nullptr == conv->common() || nullptr == conv->weight() || inputs.size() != 1) {
            return nullptr;
        }
        auto common = conv->common();
        if (common->kernels() == nullptr || common->kernels()->size() != 3 || common->strides() == nullptr
            || common->strides()->size() != 3 || common->pads() == nullptr
            || (common->pads()->size() != 3 && common->pads()->size() != 6)) {
            return nullptr;
        }
        if (common->group() > 1 || common->padMode() != PadMode_CAFFE) {
            return nullptr;
        }
        if (common->dilates() != nullptr) {
            for (int i = 0; i < (int)common->dilates()->size(); ++i) {
                if (common->dilates()->Get(i) != 1) {
                    return nullptr;
                }
            }
        }
        if (inputs[0]->dimensions() != 5 || outputs[0]->dimensions() != 5) {
            return nullptr;
        }
        const int ci = inputs[0]->length(1);
        const int co = outputs[0]->length(1);
        const size_t kvol = (size_t)common->kernels()->Get(0) * common->kernels()->Get(1) * common->kernels()->Get(2);
        if (conv->weight()->size() != (size_t)ci * co * kvol) {
            return nullptr;
        }
        if (conv->bias() != nullptr && (int)conv->bias()->size() < co) {
            return nullptr;
        }
        auto exe = new VulkanConvolution3D(bn, common, mTransposed, ci, co);
#ifdef ENABLE_VULKAN_TIME_PROFILE
        exe->setLabel(inputs[0], outputs[0]);
#endif
        if (!exe->uploadWeights(conv->weight()->data(), conv->bias() ? conv->bias()->data() : nullptr)) {
            delete exe;
            return nullptr;
        }
        return exe;
    }

private:
    bool mTransposed;
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_Convolution3D, new VulkanConvolution3DCreator(false));
    VulkanBackend::addCreator(OpType_ConvTranspose3D, new VulkanConvolution3DCreator(true));
    return true;
}();

} // namespace MNN
