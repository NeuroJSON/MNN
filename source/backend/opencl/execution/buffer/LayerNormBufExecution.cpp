//
//  LayerNormBufExecution.cpp
//  MNN
//
//  Created by MNN on 2023/07/05.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED
#include "backend/opencl/execution/buffer/LayerNormBufExecution.hpp"
#include <cstdlib>

namespace MNN {
namespace OpenCL {

LayerNormBufExecution::LayerNormBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();
    const auto* layer_norm_param = op->main_as_LayerNorm();
        mResource.reset(new LayernormResource);
    if (nullptr != layer_norm_param->axis()) {
        mResource->axis_size = layer_norm_param->axis()->size();
    }
    mResource->epsilon_ = layer_norm_param->epsilon();
    mResource->group_ = layer_norm_param->group();
    mResource->RMSNorm = layer_norm_param->useRMSNorm();
    auto bufferUnitSize = mOpenCLBackend->getPrecision() != BackendConfig::Precision_High ? sizeof(half_float::half) : sizeof(float);
    auto kernel = runtime->buildKernel("layernorm_buf", "layernorm_buf", {"-DLOCAL_SIZE=512"}, mOpenCLBackend->getPrecision());
    OPENCL_CHECK_KERNEL_CTOR(kernel);
    mResource->mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(kernel));

    mResource->has_gamma_beta_ = (layer_norm_param->gamma() && layer_norm_param->beta());
    int gammasize = 0;
    if (mResource->has_gamma_beta_) {
        MNN_ASSERT(layer_norm_param->gamma()->size() == layer_norm_param->beta()->size());
        gammasize = layer_norm_param->gamma()->size();
    }
    mResource->has_gamma_beta_ = mResource->has_gamma_beta_ || (layer_norm_param->external() && layer_norm_param->external()->size() > 1 && layer_norm_param->external()->data()[1] > 0);
    if (mResource->has_gamma_beta_ && gammasize == 0) {
        gammasize = layer_norm_param->external()->data()[1] / sizeof(float);
    }
        
    auto staticMapAlloc = mOpenCLBackend->getStaticAllocatorMMap();
    if(mResource->has_gamma_beta_){
        {
            auto error = CL_SUCCESS;
            int size = gammasize;
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mGammaBuffer = staticMapAlloc.get()->allocBuffer(ALIGN_UP4(size) * bufferUnitSize);
            }else{
                mResource->mGammaBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ALIGN_UP4(size) * bufferUnitSize));
            }
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap <= 1){
                auto GammaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mGammaBuffer.get()), true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr, nullptr, &error);
                const float* gamma_data = layer_norm_param->gamma()->data();
                if(GammaPtrCL != nullptr && error == CL_SUCCESS){
                    if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High){
                        for (int i = 0; i < size; i++)
                        {
                            ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(gamma_data[i]);
                        }
                        for(int i=size; i<ALIGN_UP4(size); i++) {
                            ((half_float::half*)GammaPtrCL)[i] = (half_float::half)(0.0f);
                        }
                    }else{
                        ::memset(GammaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                        ::memcpy(GammaPtrCL, gamma_data, size * sizeof(float));
                    }
                }else{
                    MNN_ERROR("Map error GammaPtrCL == nullptr \n");
                }
                mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mResource->mGammaBuffer.get(), GammaPtrCL);
            }
        }
        {
            auto error = CL_SUCCESS;
            int size = gammasize;
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap && staticMapAlloc != nullptr){
                mResource->mBetaBuffer = staticMapAlloc.get()->allocBuffer(ALIGN_UP4(size) * bufferUnitSize);
            }else{
                mResource->mBetaBuffer.reset(new cl::Buffer(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, ALIGN_UP4(size) * bufferUnitSize));
            }
            if(mOpenCLBackend->getRuntime()->hint().useCachedMmap <= 1){
                auto BetaPtrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(*(mResource->mBetaBuffer.get()), true, CL_MAP_WRITE, 0, ALIGN_UP4(size) * bufferUnitSize, nullptr, nullptr, &error);
                const float* beta_data = layer_norm_param->beta()->data();
                if(BetaPtrCL != nullptr && error == CL_SUCCESS){
                    if(mOpenCLBackend->getPrecision() != BackendConfig::Precision_High){
                        for (int i = 0; i < size; i++)
                        {
                            ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(beta_data[i]);
                        }
                        for(int i=size; i<ALIGN_UP4(size); i++) {
                            ((half_float::half*)BetaPtrCL)[i] = (half_float::half)(0.0f);
                        }
                    }else{
                        ::memset(BetaPtrCL, 0, ALIGN_UP4(size) * sizeof(float));
                        ::memcpy(BetaPtrCL, beta_data, size * sizeof(float));
                    }
                }else{
                    MNN_ERROR("Map error BetaPtrCL == nullptr \n");
                }
                mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(*mResource->mBetaBuffer.get(), BetaPtrCL);
            }
        }
    }
}

LayerNormBufExecution::LayerNormBufExecution(std::shared_ptr<LayernormResource> resource, const Op* op, Backend* backend): CommonExecution(backend, op) {
    mResource = resource;
    mOpenCLBackend = (OpenCLBackend *)backend;
}

bool LayerNormBufExecution::onClone(Backend *bn, const Op *op, Execution **dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new LayerNormBufExecution(mResource, op, bn);
    return true;
}

int LayerNormBufExecution::getLocalSize(int size, int maxGroupSize){
    int local_size = 1;
    while(local_size * 2 <= maxGroupSize && local_size * 2 <= size){
        local_size *= 2;
    }
    return local_size;
}

ErrorCode LayerNormBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    Tensor *input  = inputs[0];
    Tensor *output = outputs[0];
    auto runtime = ((OpenCLBackend *)backend())->getOpenCLRuntime();
    auto MaxLocalSize = std::min(std::min(runtime->getMaxWorkItemSizes()[0], mResource->mMaxWorkGroupSize), (uint32_t)256);

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);

    int rank = inputs.at(0)->dimensions();
    int outter_size = 1;
    int inner_size = 1;
    for (int i = 0; i < rank - mResource->axis_size; ++i) {
        outter_size *= inputs.at(0)->length(i);
    }
    for (int i = rank - mResource->axis_size; i < rank; ++i) {
        inner_size *= inputs.at(0)->length(i);
    }

    if (mResource->group_ > 1) {
        outter_size = inputs[0]->length(0) * mResource->group_;
        inner_size = 1;
        for (int i = 1; i < rank; i++) {
            inner_size *= inputs[0]->length(i);
        }
        inner_size /= mResource->group_;
    }
    
    // One work-group per row leaves a large GPU mostly idle when there are few,
    // very long rows (InstanceNorm on a 3D volume: 32 rows of 4.7M values at
    // SIAM's full resolution). Split those rows across work-groups instead.
    // MNN_LAYERNORM_NOSPLIT=1 keeps the single-kernel path.
    static int s_nosplit = -1;
    if (s_nosplit < 0) {
        const char* e = std::getenv("MNN_LAYERNORM_NOSPLIT");
        s_nosplit = (e && e[0] == '1') ? 1 : 0;
    }
    if (s_nosplit == 0 && inner_size % 4 == 0 && inner_size >= 65536) {
        return encodeSplit(input, output, outter_size, inner_size);
    }

    int local_size = getLocalSize(inner_size / 4, MaxLocalSize);
    std::set<std::string> buildOptions;
    buildOptions.emplace("-DLOCAL_SIZE=" + std::to_string(local_size));
    if(mResource->RMSNorm){
        buildOptions.emplace("-DRMSNORM");
    }
    if(mResource->has_gamma_beta_){
        buildOptions.emplace("-DGAMMA_BETA");
    }
    if(inner_size % 4 != 0){
        buildOptions.emplace("-DPACK_LEAVE");
    }
    
    unit.kernel = runtime->buildKernel("layernorm_buf", "layernorm_buf", buildOptions, mOpenCLBackend->getPrecision());
    mGWS = {static_cast<uint32_t>(local_size), static_cast<uint32_t>(outter_size)};
    mLWS = {static_cast<uint32_t>(local_size), 1};

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGWS[0]);
    ret |= unit.kernel->get().setArg(idx++, mGWS[1]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, static_cast<int32_t>(inner_size));
    if(mResource->has_gamma_beta_){
        ret |= unit.kernel->get().setArg(idx++, *mResource->mGammaBuffer.get());
        ret |= unit.kernel->get().setArg(idx++, *mResource->mBetaBuffer.get());
    }
    ret |= unit.kernel->get().setArg(idx++, mResource->epsilon_);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LayerNormBufExecution");
    mOpenCLBackend->recordKernel2d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1]};
    unit.localWorkSize = {mLWS[0], mLWS[1]};

    return NO_ERROR;

}

ErrorCode LayerNormBufExecution::encodeSplit(Tensor* input, Tensor* output, int outter_size, int inner_size) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    const int kLocal  = 256;              // SPLIT_LOCAL: power of two
    const int inside4 = inner_size / 4;
    const int chunk4  = kLocal * 8;       // float4s per work-group (8 per work-item)
    const int chunks  = UP_DIV(inside4, chunk4);

    const size_t statsBytes = static_cast<size_t>(outter_size) * chunks * 3 * sizeof(float);
    const size_t rowBytes   = static_cast<size_t>(outter_size) * 2 * sizeof(float);
    if (nullptr == mSplitStats || mSplitStatsBytes < statsBytes) {
        mSplitStats.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE, statsBytes));
        mSplitStatsBytes = statsBytes;
    }
    if (nullptr == mSplitRow || mSplitRowBytes < rowBytes) {
        mSplitRow.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE, rowBytes));
        mSplitRowBytes = rowBytes;
    }

    std::set<std::string> options = {"-DSPLIT_LOCAL=" + std::to_string(kLocal)};
    if (mResource->RMSNorm) {
        options.emplace("-DRMSNORM");
    }
    if (mResource->has_gamma_beta_) {
        options.emplace("-DGAMMA_BETA");
    }
    mUnits.resize(3);
    cl_int ret = CL_SUCCESS;

    auto& stats = mUnits[0];
    stats.kernel = runtime->buildKernel("layernorm_buf", "layernorm_split_stats", options, mOpenCLBackend->getPrecision());
    if (nullptr == stats.kernel) {
        return NOT_SUPPORT;
    }
    uint32_t idx = 0;
    ret |= stats.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= stats.kernel->get().setArg(idx++, *mSplitStats);
    ret |= stats.kernel->get().setArg(idx++, inside4);
    ret |= stats.kernel->get().setArg(idx++, chunk4);
    ret |= stats.kernel->get().setArg(idx++, chunks);
    std::vector<uint32_t> gws0 = {static_cast<uint32_t>(kLocal * chunks), static_cast<uint32_t>(outter_size)};
    std::vector<uint32_t> lws0 = {static_cast<uint32_t>(kLocal), 1};
    mOpenCLBackend->recordKernel2d(stats.kernel, gws0, lws0);
    stats.globalWorkSize = {gws0[0], gws0[1]};
    stats.localWorkSize  = {lws0[0], lws0[1]};

    auto& fin = mUnits[1];
    fin.kernel = runtime->buildKernel("layernorm_buf", "layernorm_split_finalize", options, mOpenCLBackend->getPrecision());
    if (nullptr == fin.kernel) {
        return NOT_SUPPORT;
    }
    idx = 0;
    ret |= fin.kernel->get().setArg(idx++, *mSplitStats);
    ret |= fin.kernel->get().setArg(idx++, *mSplitRow);
    ret |= fin.kernel->get().setArg(idx++, chunks);
    ret |= fin.kernel->get().setArg(idx++, outter_size);
    ret |= fin.kernel->get().setArg(idx++, mResource->epsilon_);
    std::vector<uint32_t> gws1 = {static_cast<uint32_t>(ROUND_UP(outter_size, 64)), 1};
    std::vector<uint32_t> lws1 = {64, 1};
    mOpenCLBackend->recordKernel2d(fin.kernel, gws1, lws1);
    fin.globalWorkSize = {gws1[0], gws1[1]};
    fin.localWorkSize  = {lws1[0], lws1[1]};

    auto& norm = mUnits[2];
    norm.kernel = runtime->buildKernel("layernorm_buf", "layernorm_split_norm", options, mOpenCLBackend->getPrecision());
    if (nullptr == norm.kernel) {
        return NOT_SUPPORT;
    }
    const uint32_t g0 = static_cast<uint32_t>(inside4), g1 = static_cast<uint32_t>(outter_size);
    idx = 0;
    ret |= norm.kernel->get().setArg(idx++, g0);
    ret |= norm.kernel->get().setArg(idx++, g1);
    ret |= norm.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= norm.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= norm.kernel->get().setArg(idx++, *mSplitRow);
    if (mResource->has_gamma_beta_) {
        ret |= norm.kernel->get().setArg(idx++, *mResource->mGammaBuffer.get());
        ret |= norm.kernel->get().setArg(idx++, *mResource->mBetaBuffer.get());
    }
    ret |= norm.kernel->get().setArg(idx++, inside4);
    MNN_CHECK_CL_SUCCESS(ret, "setArg LayerNormBufExecution split");
    std::vector<uint32_t> gws2 = {g0, g1};
    std::vector<uint32_t> lws2 = {256, 1};
    mOpenCLBackend->recordKernel2d(norm.kernel, gws2, lws2);
    norm.globalWorkSize = {gws2[0], gws2[1]};
    norm.localWorkSize  = {lws2[0], lws2[1]};
    return NO_ERROR;
}

class LayerNormBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~LayerNormBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
		for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        const auto* layer_norm_param = op->main_as_LayerNorm();
        OPENCL_CREATOR_CHECK(new LayerNormBufExecution(inputs, op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(LayerNormBufCreator, OpType_LayerNorm, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif /* MNN_OPENCL_BUFFER_CLOSED */
