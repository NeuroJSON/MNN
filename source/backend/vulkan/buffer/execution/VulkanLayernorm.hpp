//
//  VulkanLayernorm.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanLayernorm_hpp
#define VulkanLayernorm_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include "VulkanBasicExecution.hpp"

namespace MNN {

class VulkanLayernorm : public VulkanBasicExecution {
public:
    VulkanLayernorm(const Op* op, Backend* bn, Tensor * tensor);
    virtual ~VulkanLayernorm();
    virtual ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;
    virtual bool onClone(Backend* bn, const Op* op, VulkanBasicExecution** dst) override;

private:
    VulkanLayernorm(Backend* bn, const VulkanLayernorm* src);
    std::shared_ptr<VulkanBuffer> mParam;
    std::shared_ptr<Tensor> mGamma;
    std::shared_ptr<Tensor> mBias;
    const VulkanPipeline* mPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDesSet;
    const VulkanPipeline* mOptPipeline = nullptr;
    std::shared_ptr<VulkanLayout::DescriptorSet> mOptDesSet;
    std::string mKey;
    std::string mOptKey;
    std::vector<VkDescriptorType> mDesTypes;
    float mEps;
    bool mHasScale = false;
    bool mUseRMSNorm = false;
    int mGroup = 0;
    int mAxisSize = 0;
    bool mFP16{false};
    // Split-row path (glsl/norm_split.comp) for few, very long rows.
    ErrorCode encodeSplit(const Tensor* input, const Tensor* output, int inside, int outside,
                          const VulkanCommandPool::Buffer* cmdBuffer);
    std::shared_ptr<VulkanLayout::DescriptorSet> mSplitSet[3];
    std::shared_ptr<VulkanBuffer> mSplitParam;
    std::shared_ptr<VulkanBuffer> mSplitStats;  // per (row, chunk): n, mean, M2
    std::shared_ptr<VulkanBuffer> mSplitRow;    // per row: mean, 1/sqrt(var + eps)
};

} // namespace MNN

#endif /* VulkanLayernorm_hpp */