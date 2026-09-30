//
//  VulkanLayernorm.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanLayernorm.hpp"
#include "core/Macro.h"
#include <cstdlib>

namespace MNN {

struct Param {
    ivec4 size;
    float eps;
};

VulkanLayernorm::VulkanLayernorm(const Op* op, Backend* backend, Tensor * tensor) : VulkanBasicExecution(backend) {
    auto layer_norm_param = op->main_as_LayerNorm();
    auto vkbackend = static_cast<VulkanBackend*>(backend);
    if (nullptr != layer_norm_param->axis()) {
        mAxisSize = layer_norm_param->axis()->size();
    }
    mGroup = layer_norm_param->group();
    mUseRMSNorm = layer_norm_param->useRMSNorm();

    mParam = vkbackend->allocUniform();
    mEps = layer_norm_param->epsilon();

    mFP16 = tensor->getType().code == halide_type_float && vkbackend->useFP16();

    if (layer_norm_param->gamma() && layer_norm_param->beta()) {
        mHasScale = true;
        int size = layer_norm_param->gamma()->size();
        auto prepareParam = [&](std::shared_ptr<Tensor>& paramTensor, const float* sourceData, const char* errorName) {
            paramTensor.reset(Tensor::createDevice<float>({size}));
            auto status = backend->onAcquireBuffer(paramTensor.get(), Backend::STATIC);
            if (!status) {
                MNN_ERROR("Out of memory when %s is acquired in LayerNorm.\n", errorName);
                return false;
            }
            const void * paramData;
            std::vector<int16_t> paramFP16;
            if (mFP16) {
                paramFP16.resize(size);
                FLOAT_TO_HALF(sourceData, paramFP16.data(), size);
                paramData = paramFP16.data();
            } else {
                paramData = (const void *) sourceData;
            }
            auto paramBuffer = vkbackend->getBuffer(paramTensor.get());
            vkbackend->copyToGPUBuffer(paramData, std::get<0>(paramBuffer), std::get<1>(paramBuffer), std::get<2>(paramBuffer));
            return true;
        };

        if (!prepareParam(mGamma, layer_norm_param->gamma()->data(), "gamma")) {
            return;
        }

        if (layer_norm_param->beta()->size() != size) {
            MNN_ERROR("Size of gamma and beta are not match in LayerNorm.\n");
            return;
        }

        if (!prepareParam(mBias, layer_norm_param->beta()->data(), "beta")) {
            return;
        }
    }

    mKey = "glsl_norm_";
    if (!mHasScale) {
        mDesTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
    } else {
        mDesTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        mKey += "LAYERNORM_SCALE_";
    }
    if (mFP16) {
        mKey += "FP16_";
    }
    mKey += "comp";
    std::vector<uint32_t> spec = {mUseRMSNorm ? 1u : 0u};
    mPipeline = vkbackend->getPipeline(mKey, mDesTypes, {}, spec);
    mDesSet.reset(mPipeline->createSet());

    mOptKey = "glsl_norm_opt_";
    if (mHasScale) {
        mOptKey += "LAYERNORM_SCALE_";
    }
    if (mFP16) {
        mOptKey += "FP16_";
    }
    mOptKey += "comp";
    mOptPipeline = vkbackend->getPipeline(mOptKey, mDesTypes, {}, spec);
    mOptDesSet.reset(mOptPipeline->createSet());
}

VulkanLayernorm::VulkanLayernorm(Backend* bn, const VulkanLayernorm* src)
    : VulkanBasicExecution(bn)
    , mEps(src->mEps)
    , mHasScale(src->mHasScale)
    , mUseRMSNorm(src->mUseRMSNorm)
    , mGroup(src->mGroup)
    , mAxisSize(src->mAxisSize)
    , mFP16(src->mFP16)
    , mKey(src->mKey)
    , mOptKey(src->mOptKey)
    , mDesTypes(src->mDesTypes) {
    auto vkbackend = static_cast<VulkanBackend*>(bn);
    mParam = vkbackend->allocUniform();
    mGamma = src->mGamma;
    mBias = src->mBias;
    std::vector<uint32_t> spec = {mUseRMSNorm ? 1u : 0u};
    mPipeline = vkbackend->getPipeline(mKey, mDesTypes, {}, spec);
    mDesSet.reset(mPipeline->createSet());
    mOptPipeline = vkbackend->getPipeline(mOptKey, mDesTypes, {}, spec);
    mOptDesSet.reset(mOptPipeline->createSet());
}

VulkanLayernorm::~VulkanLayernorm() {
    auto vkbackend = static_cast<VulkanBackend*>(backend());
    vkbackend->recycleUniform(mParam);
    if (nullptr != mSplitParam) {
        vkbackend->recycleUniform(mSplitParam);
    }
}

bool VulkanLayernorm::onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto res = new VulkanLayernorm(bn, this);
    *dst = res;
    return true;
}

ErrorCode VulkanLayernorm::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const VulkanCommandPool::Buffer* cmdBuffer) {
    // set param
    auto vkBn = (VulkanBackend*)backend();
    auto outside = 1;
    auto inside = 1;
    int rank = inputs.at(0)->dimensions();
    if (mGroup > 1) {
        outside = inputs.at(0)->length(0) * mGroup;
        for (int i = 1; i < rank; i++) {
            inside *= inputs.at(0)->length(i);
        }
        inside /= mGroup;
    } else {
        for (int i = 0; i < rank - mAxisSize; ++i) {
            outside *= inputs.at(0)->length(i);
        }
        for (int i = rank - mAxisSize; i < rank; ++i) {
            inside *= inputs.at(0)->length(i);
        }
    }
    auto param = reinterpret_cast<Param*>(mParam->map());
    param->size[0] = inside;
    param->size[1] = outside;
    param->size[2] = mUseRMSNorm ? 1 : 0;
    param->size[3] = outside;
    param->eps = mEps;
    mParam->unmap();
    auto inputTensor = vkBn->getBuffer(inputs[0]);
    auto outputTensor = vkBn->getBuffer(outputs[0]);
    // One work-group per row (norm_opt) leaves a large GPU mostly idle when
    // there are few, very long rows (InstanceNorm on a 3D volume: 32 rows of
    // 4.7M values at SIAM's full resolution); split those rows across
    // work-groups. MNN_LAYERNORM_NOSPLIT=1 keeps the per-row kernels.
    static int s_nosplit = -1;
    if (s_nosplit < 0) {
        const char* e = ::getenv("MNN_LAYERNORM_NOSPLIT");
        s_nosplit = (e && e[0] == '1') ? 1 : 0;
    }
    if (s_nosplit == 0 && (inside & 3) == 0 && inside >= 65536
        && (uint32_t)outside <= vkBn->getDevice().proty().limits.maxComputeWorkGroupCount[1]) {
        const ErrorCode code = encodeSplit(inputs[0], outputs[0], inside, outside, cmdBuffer);
        if (NO_ERROR == code) {
            return NO_ERROR;
        }
    }
    auto maxGroupCountX = (int)vkBn->getDevice().proty().limits.maxComputeWorkGroupCount[0];
    // LLM-oriented fast path: 1 workgroup per row (outside), parallel reduce over inside.
    // Requires inside % 4 == 0; fallback when dispatch count might exceed device limits.
    bool useOpt = (outside <= maxGroupCountX) && ((inside & 3) == 0);
    if (useOpt) {
        mOptDesSet->writeBuffer(outputTensor, 0);
        mOptDesSet->writeBuffer(inputTensor, 1);
        mOptDesSet->writeBuffer(mParam->buffer(), 2, mParam->size());
        if (mHasScale) {
            mOptDesSet->writeBuffer(vkBn->getBuffer(mGamma.get()), 3);
            mOptDesSet->writeBuffer(vkBn->getBuffer(mBias.get()), 4);
        }
        mOptPipeline->bind(cmdBuffer->get(), mOptDesSet->get());
        vkCmdDispatch(cmdBuffer->get(), outside, 1, 1);
    } else {
        mDesSet->writeBuffer(outputTensor, 0);
        mDesSet->writeBuffer(inputTensor, 1);
        mDesSet->writeBuffer(mParam->buffer(), 2, mParam->size());
        if (mHasScale) {
            mDesSet->writeBuffer(vkBn->getBuffer(mGamma.get()), 3);
            mDesSet->writeBuffer(vkBn->getBuffer(mBias.get()), 4);
        }
        mPipeline->bind(cmdBuffer->get(), mDesSet->get());
        vkCmdDispatch(cmdBuffer->get(), UP_DIV(outside, 64), 1, 1);
    }

    return NO_ERROR;
}

ErrorCode VulkanLayernorm::encodeSplit(const Tensor* input, const Tensor* output, int inside, int outside,
                                       const VulkanCommandPool::Buffer* cmdBuffer) {
    auto vkBn = static_cast<VulkanBackend*>(backend());
    const int kLocal  = 256;              // matches the shared array in norm_split.comp
    const int inside4 = inside / 4;
    const int chunk4  = kLocal * 8;       // vec4s per work-group (8 per invocation)
    const int chunks  = UP_DIV(inside4, chunk4);
    const auto& limits = vkBn->getDevice().proty().limits;
    if ((uint32_t)chunks > limits.maxComputeWorkGroupCount[0]
        || (uint32_t)UP_DIV(inside4, kLocal) > limits.maxComputeWorkGroupCount[0]) {
        return NOT_SUPPORT;
    }
    auto ensure = [&](std::shared_ptr<VulkanBuffer>& buf, size_t bytes) {
        if (nullptr == buf || buf->size() < bytes) {
            buf.reset(new VulkanBuffer(vkBn->getMemoryPool(), false, bytes, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_SHARING_MODE_EXCLUSIVE, 0));
        }
        return buf->valid();
    };
    if (!ensure(mSplitStats, (size_t)outside * chunks * 3 * sizeof(float))
        || !ensure(mSplitRow, (size_t)outside * 2 * sizeof(float))) {
        return OUT_OF_MEMORY;
    }
    if (nullptr == mSplitParam) {
        mSplitParam = vkBn->allocUniform();
    }
    {
        auto param = reinterpret_cast<Param*>(mSplitParam->map());
        param->size[0] = inside4;
        param->size[1] = outside;
        param->size[2] = chunk4;
        param->size[3] = chunks;
        param->eps = mEps;
        mSplitParam->unmap();
    }
    std::vector<VkDescriptorType> types(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    types[2] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    std::string key = "glsl_norm_split_";
    if (mHasScale) {
        types.push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        types.push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        key += "LAYERNORM_SCALE_";
    }
    if (mFP16) {
        key += "FP16_";
    }
    key += "comp";
    const uint32_t groups[3][3] = {{(uint32_t)chunks, (uint32_t)outside, 1},
                                   {(uint32_t)UP_DIV(outside, kLocal), 1, 1},
                                   {(uint32_t)UP_DIV(inside4, kLocal), (uint32_t)outside, 1}};
    for (int stage = 0; stage < 3; ++stage) {
        auto pipeline = vkBn->getPipeline(key, types, {(uint32_t)kLocal, 1, 1},
                                          {mUseRMSNorm ? 1u : 0u, (uint32_t)stage});
        if (nullptr == pipeline) {
            return NOT_SUPPORT;
        }
        mSplitSet[stage].reset(pipeline->createSet());
        auto& set = mSplitSet[stage];
        set->writeBuffer(vkBn->getBuffer(output), 0);
        set->writeBuffer(vkBn->getBuffer(input), 1);
        set->writeBuffer(mSplitParam->buffer(), 2, mSplitParam->size());
        set->writeBuffer(mSplitStats->buffer(), 3, mSplitStats->size());
        set->writeBuffer(mSplitRow->buffer(), 4, mSplitRow->size());
        if (mHasScale) {
            set->writeBuffer(vkBn->getBuffer(mGamma.get()), 5);
            set->writeBuffer(vkBn->getBuffer(mBias.get()), 6);
        }
        pipeline->bind(cmdBuffer->get(), set->get());
        vkCmdDispatch(cmdBuffer->get(), groups[stage][0], groups[stage][1], groups[stage][2]);
        // The next stage reads what this one wrote.
        if (stage == 0) {
            cmdBuffer->barrierSource(mSplitStats->buffer(), 0, mSplitStats->size());
        } else if (stage == 1) {
            cmdBuffer->barrierSource(mSplitRow->buffer(), 0, mSplitRow->size());
        }
    }
    return NO_ERROR;
}

class VulkanLayernormCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op, Backend* bn) const override {
        return new VulkanLayernorm(op, bn, inputs[0]);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_LayerNorm, new VulkanLayernormCreator);
    return true;
}();

} // namespace MNN