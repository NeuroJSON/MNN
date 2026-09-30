//
//  VulkanBackend.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanBackend.hpp"
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>
#include "core/Execution.hpp"
#include "core/Macro.h"
#include <MNN/Tensor.hpp>
#include "core/TensorUtils.hpp"
#include "component/VulkanDevice.hpp"
#include "component/VulkanInstance.hpp"
#include "execution/VulkanBasicExecution.hpp"
//#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
// #define MNN_OP_SUPPORT_LOG

#ifdef ENABLE_VULKAN_TIME_PROFILE
#include <chrono>
#endif
//#define MNN_VULKAN_DUMP_MEMORY_USAGE


namespace MNN {

static std::map<OpType, VulkanBackend::Creator*>* gCreator = nullptr;

// Creator
static inline std::map<OpType, VulkanBackend::Creator*>* getCreatorMap() {
    if (nullptr == gCreator) {
        gCreator = new std::map<OpType, VulkanBackend::Creator*>();
    }
    return gCreator;
}

template<typename T0, typename T1>
void _copy(const T0* src, T1* dst, size_t size) {
    for (int i=0; i<size; ++i) {
        dst[i] = src[i];
    }
}

#ifndef MNN_USE_ARMV82

void _VKFloatToHalf(const float* src, int16_t* dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        ((half_float::half *)dst)[i] = (half_float::half)(src[i]);
    }
    return;
}

void _VKHalfToFloat(const int16_t* src, float* dst, size_t size) {
    const size_t batchSize = 8;
    std::vector<half_float::half> halfBatch(batchSize);

    for (size_t i = 0; i < size; i += batchSize) {
        size_t currentBatchSize = std::min(batchSize, size - i);

        ::memcpy(halfBatch.data(), &(src[i]), currentBatchSize * sizeof(int16_t));

        for (size_t j = 0; j < currentBatchSize; ++j) {
            dst[i + j] = static_cast<float>(halfBatch[j]);
        }
    }

    return;
}

#endif

// Host <-> staging-buffer copies of large tensors (e.g. 3D segmentation
// logits, hundreds of MB per inference) are bound by a single core's memcpy /
// fp16<->fp32 conversion throughput, several times below what the memory
// system sustains. Split them across a few threads; small copies stay serial.
// fn(start, count) processes elements [start, start + count).
template <typename Fn>
static void _parallelChunks(size_t count, size_t elemBytes, Fn fn) {
    const size_t kMinParallelBytes = 16 * 1024 * 1024;
    unsigned int n = std::thread::hardware_concurrency();
    n = std::min(n == 0 ? 1u : n, 8u);
    if (count * elemBytes < kMinParallelBytes || n < 2) {
        fn((size_t)0, count);
        return;
    }
    const size_t chunk = UP_DIV(count, (size_t)n * 16) * 16;
    std::vector<std::thread> workers;
    for (unsigned int i = 1; i < n && i * chunk < count; ++i) {
        const size_t start = i * chunk;
        const size_t len = std::min(chunk, count - start);
        workers.emplace_back([=]() { fn(start, len); });
    }
    fn((size_t)0, std::min(chunk, count));
    for (auto& w : workers) {
        w.join();
    }
}

static void _parallelMemcpy(void* dst, const void* src, size_t bytes) {
    _parallelChunks(bytes, 1, [=](size_t start, size_t len) {
        ::memcpy((uint8_t*)dst + start, (const uint8_t*)src + start, len);
    });
}

static void _copyBufferToTensor(const Tensor* dest, const VulkanBuffer* source, size_t offset, bool half2float = false) {
    auto sourcePtr   = (const float*)source->map(offset);
    if (half2float) {
        auto dstPtr = dest->host<float>();
        auto elementCount = static_cast<size_t>(dest->elementSize());
        auto halfPtr = reinterpret_cast<const int16_t*>(sourcePtr);
        _parallelChunks(elementCount, sizeof(float), [=](size_t start, size_t len) {
            HALF_TO_FLOAT(halfPtr + start, dstPtr + start, len);
        });
    } else {
        _parallelMemcpy(dest->host<float>(), sourcePtr, dest->usize());
    }
    source->unmap();
}

static void _copyTensorToBuffer(const Tensor* source, const VulkanBuffer* dest, size_t offset, bool float2half = false) {
    auto destPtr = reinterpret_cast<uint8_t*>(dest->map(offset));
    if (float2half) {
        auto srcPtr = source->host<float>();
        auto elementCount = static_cast<size_t>(source->elementSize());
        auto halfPtr = reinterpret_cast<int16_t*>(destPtr);
        _parallelChunks(elementCount, sizeof(float), [=](size_t start, size_t len) {
            FLOAT_TO_HALF(srcPtr + start, halfPtr + start, len);
        });
    } else {
        _parallelMemcpy(destPtr, source->host<float>(), source->usize());
    }
    dest->unmap();
}

VulkanBackend::VulkanBackend(const VulkanRuntime* runtime) : Backend(MNN_FORWARD_VULKAN) {
    mRuntime = runtime;
    mDirect = (mRuntime->mGpuMode & MNNGpuMode::MNN_GPU_RECORD_BATCH) == 0;
    mUseFP16 = (mRuntime->mPrecision != BackendConfig::Precision_High && mRuntime->mDevice->getFP16Support());
    std::shared_ptr<BufferAllocator::Allocator> allocReal = BufferAllocator::Allocator::createRecurse(runtime->mBufferPool.get());
    mDynamicBufferPool.resize(2);
    mDynamicBufferPool[0].reset(new EagerBufferAllocator(allocReal, mRuntime->mDevice->proty().limits.nonCoherentAtomSize));
    mCurrentDynamicBufferPool = mDynamicBufferPool[0].get();

    auto& dev              = device();
    mFence                 = std::make_shared<VulkanFence>(dev);
#ifdef ENABLE_VULKAN_TIME_PROFILE
    mTimeProfiler = std::make_shared<VulkanTimeProfiler>(dev);
#endif
    std::string deviceName = dev.proty().deviceName;
    if(deviceName.find("Apple") != std::string::npos){
        mUseAutoTune = false;
    }
    mCmdBufferForCopy.reset(runtime->mCmdPool->allocBuffer());
}

VulkanBackend::~VulkanBackend() {
    /*keep release order*/
    mCurrentIndirectSegment = nullptr;
    mIndirectSegments.clear();
    mCmdBuffers.clear();
    mFence = nullptr;
}
void VulkanBackend::pushCommand(VkCommandBuffer buffer) const {
    mCmdBuffers.emplace_back(buffer);
}

std::shared_ptr<VulkanCommandPool::Buffer> VulkanBackend::acquireIndirectSegmentForRecord() {
    MNN_ASSERT(!mDirect);
    if (nullptr == mCurrentIndirectSegment.get()) {
        mCurrentIndirectSegment.reset(mRuntime->mCmdPool->allocBuffer());
        mCurrentIndirectSegment->begin(0);
        mCurrentIndirectSegmentOpCount = 0;
    }
    return mCurrentIndirectSegment;
}

void VulkanBackend::finishIndirectRecordedOp() {
    if (mDirect) {
        return;
    }
    MNN_ASSERT(nullptr != mCurrentIndirectSegment.get());
    ++mCurrentIndirectSegmentOpCount;
    if (mCurrentIndirectSegmentOpCount >= kIndirectSegmentOpLimit) {
        _sealIndirectSegment();
    }
}

const VulkanPipeline* VulkanBackend::getPipeline(const std::string& key, const std::vector<VkDescriptorType>& types,
                                                 const std::vector<uint32_t>& localSize,
                                                 const std::vector<uint32_t>& specConstants) const {
    return mRuntime->mPipelineFactory->getPipeline(key, types, localSize, specConstants);
}

SharedPtr<VulkanPipeline> VulkanBackend::getPrivatePipeline(const std::string& key, const std::vector<VkDescriptorType>& types, const std::vector<uint32_t>& specConstants) {
    return mRuntime->mPipelineFactory->getPrivatePipeline(key, types, specConstants);
}

void VulkanBackend::onResizeBegin() {
#ifdef ENABLE_VULKAN_TIME_PROFILE
    if (mTimeProfiler) {
        mTimeProfiler->reset();
    }
#endif
    if (!mDirect) {
        _resetIndirectSegments();
    }
}
ErrorCode VulkanBackend::onResizeEnd() {
    if (!mDirect) {
        _sealIndirectSegment();
    }
    mHostBuffer.reset();
    return NO_ERROR;
}
class VulkanMemRelease : public Backend::MemObj {
public:
    VulkanMemRelease(BufferAllocator* allocator, MemChunk points, int size) {
        mPoint = std::move(points);
        mAllocator = allocator;
        mSize = size;
    }
    virtual ~ VulkanMemRelease() {
        mAllocator->free(mPoint);
    }
    inline int getSize() const {
        return mSize;
    }
    inline MemChunk points() const {
        return mPoint;
    }
private:
    BufferAllocator* mAllocator;
    MemChunk mPoint;
    int mSize;
};
VULKAN_TENSOR VulkanBackend::getBuffer(const Tensor* tensor) const {
    auto b = getTensorBuffer(tensor);
    return std::make_tuple(b.first->buffer(), getTensorSize(tensor), b.second);
}

std::pair<const VulkanBuffer*, size_t> VulkanBackend::getTensorBuffer(const Tensor* tensor) const {
    auto mem = (VulkanBuffer*)(tensor->deviceId());
    MNN_ASSERT(nullptr != mem);
    return std::make_pair(mem, TensorUtils::getDescribeOrigin(tensor)->offset);
}

size_t VulkanBackend::getTensorSize(const Tensor* tensor) const {
    size_t alignElementSize = (size_t) UP_DIV(tensor->elementSize(), 4) * 4;
    size_t bytes = ((tensor->getType().code == halide_type_float) && mUseFP16) ? sizeof(uint16_t) : sizeof(float);
    size_t size = alignElementSize * bytes;
    return size;
}

Backend::MemObj* VulkanBackend::onAcquire(const Tensor* tensor, StorageType storageType) {
    MNN_ASSERT(tensor->getType().code == halide_type_float || tensor->getType().code == halide_type_int);
    //FUNC_PRINT_ALL(tensor, p);
    auto alignSize = getTensorSize(tensor);
    auto MTensor     = const_cast<Tensor*>(tensor);
    auto des = TensorUtils::getDescribeOrigin(tensor);
    if (Backend::STATIC == storageType) {
        auto newBuffer = mRuntime->mBufferPool->alloc(alignSize);
        auto mem = new VulkanMemRelease(mRuntime->mBufferPool.get(), newBuffer, alignSize);
        MTensor->buffer().device = (uint64_t)(newBuffer.first);
        des->offset = newBuffer.second;
        return mem;
    }
    bool seperate  = storageType == Backend::DYNAMIC_SEPERATE;
    auto newBuffer = mCurrentDynamicBufferPool->alloc(alignSize, seperate);
    auto mem = new VulkanMemRelease(mCurrentDynamicBufferPool, newBuffer, alignSize);
    MTensor->buffer().device = (uint64_t)(newBuffer.first);
    des->offset = newBuffer.second;
    return mem;
}
bool VulkanBackend::onSelectDynamicAllocator(int index, int maxIndex) {
    if (maxIndex > 2 || index >= 2 || index < 0) {
        return false;
    }
    if (mDynamicBufferPool[1].get() == nullptr) {
        std::shared_ptr<BufferAllocator::Allocator> allocReal = BufferAllocator::Allocator::createRecurse(mRuntime->mBufferPool.get());
        mDynamicBufferPool[1].reset(new EagerBufferAllocator(allocReal, mRuntime->mDevice->proty().limits.nonCoherentAtomSize));
    }
    mCurrentDynamicBufferPool = mDynamicBufferPool[index].get();
    return true;
}

std::shared_ptr<VulkanBuffer> VulkanBackend::allocUniform(const void* src, int size) {
    auto rt = const_cast<VulkanRuntime*>(mRuntime);
    return rt->allocUniform(src, size);
}
void VulkanBackend::recycleUniform(std::shared_ptr<VulkanBuffer> buffer) {
    auto rt = const_cast<VulkanRuntime*>(mRuntime);
    rt->recycleUniform(buffer);
}

bool VulkanBackend::onClearBuffer() {
    mCurrentDynamicBufferPool->release(false);
    _resetIndirectSegments();
    return true;
}
Execution* VulkanBackend::onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                   const MNN::Op* op) {
    auto creator = getCreatorMap();
    auto iter    = creator->find(op->type());
    std::string name = "";
    if (nullptr != op->name()) {
        name = op->name()->str();
    }
    if (iter == creator->end()) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_PRINT("Vulkan don't support %d, %s: %s\n", op->type(), EnumNameOpType(op->type()),
                name.c_str());
#endif
        return nullptr;
    }
    std::shared_ptr<VulkanBasicExecution> originExecution ((VulkanBasicExecution*)iter->second->onCreate(inputs, outputs, op, this));
    if (nullptr == originExecution) {
#ifdef MNN_OP_SUPPORT_LOG
        MNN_ERROR("Vulkan don't support for %s, type=%s, Special case\n", name.c_str(), EnumNameOpType(op->type()));
#endif
        return nullptr;
    }

#ifdef ENABLE_VULKAN_TIME_PROFILE
    if (originExecution->getName() == kVulkanTimeProfileDefaultExecutionName) {
        originExecution->setName(EnumNameOpType(op->type()));
    }
#endif

    if (mDirect) {
        return new VulkanBasicExecutionDirect(originExecution);
    }
    return new VulkanBasicExecutionInDirect(originExecution);
}

void VulkanBackend::onExecuteBegin() const {
}

void VulkanBackend::onExecuteEnd() const {
    if (!mDirect) {
        mCmdBuffers.reserve(mCmdBuffers.size() + mIndirectSegments.size());
        for (auto& segment : mIndirectSegments) {
            mCmdBuffers.push_back(segment->get());
        }
    }
#ifdef ENABLE_VULKAN_TIME_PROFILE
    auto startTime = std::chrono::high_resolution_clock::now();
    _finish();
    auto endTime = std::chrono::high_resolution_clock::now();
    float totalTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / (1e6f);
    if (mTimeProfiler) {
        MNN_PRINT("\n=============== Vulkan Time Profiling (Begin) ===============\n");
        mTimeProfiler->printTimeProfile();
        MNN_PRINT("Total time calculated by CPU is %6.2f ms.\n", totalTime);
        MNN_PRINT("\n================ Vulkan Time Profiling (End) ================\n");
    }
#else
    _finish();
#endif
}

void VulkanBackend::finish() {
    _finish();
}

void VulkanBackend::_finish() const {
    if (mCmdBuffers.empty()) {
        return;
    }
    VkSubmitInfo submit_info = {/* .sType                = */ VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                /* .pNext                = */ nullptr,
                                /* .waitSemaphoreCount   = */ 0,
                                /* .pWaitSemaphores      = */ nullptr,
                                /* .pWaitDstStageMask    = */ nullptr,
                                /* .commandBufferCount   = */ (uint32_t)mCmdBuffers.size(),
                                /* .pCommandBuffers      = */ mCmdBuffers.data(),
                                /* .signalSemaphoreCount = */ 0,
                                /* .pSignalSemaphores    = */ nullptr};
    auto fenceReal           = mFence->get();
    mFence->reset();
    CALL_VK(vkQueueSubmit(device().acquireDefaultDevQueue(), 1, &submit_info, fenceReal));

    auto res = mFence->wait();
    MNN_VK_CHECK(res);
    mCmdBuffers.clear();
}

void VulkanBackend::_resetIndirectSegments() {
    mCurrentIndirectSegment = nullptr;
    mIndirectSegments.clear();
    mCurrentIndirectSegmentOpCount = 0;
}

void VulkanBackend::_sealIndirectSegment() {
    if (nullptr == mCurrentIndirectSegment.get()) {
        return;
    }
    if (mCurrentIndirectSegmentOpCount <= 0) {
        mCurrentIndirectSegment = nullptr;
        return;
    }
    mCurrentIndirectSegment->end();
    mIndirectSegments.emplace_back(mCurrentIndirectSegment);
    mCurrentIndirectSegment = nullptr;
    mCurrentIndirectSegmentOpCount = 0;
}

const VulkanDevice& VulkanBackend::device() const {
    return (* mRuntime->mDevice);
}
const VulkanPipelineFactory* VulkanBackend::getPipelineFactory() const {
    return mRuntime->mPipelineFactory.get();
}

static Tensor::DimensionType _convert(MNN_DATA_FORMAT format) {
    switch (format) {
        case MNN_DATA_FORMAT_NCHW:
            return Tensor::CAFFE;
        case MNN_DATA_FORMAT_NC4HW4:
            return Tensor::CAFFE_C4;
        case MNN_DATA_FORMAT_NHWC:
            return Tensor::TENSORFLOW;
        default:
            break;
    }
    return Tensor::CAFFE;
}
std::shared_ptr<VulkanBuffer> VulkanBackend::createHostBuffer(size_t size) const {
    // Staging buffers are read back by the CPU (output copies), so prefer
    // HOST_CACHED memory: plain HOST_VISIBLE is write-combined on discrete GPUs
    // and CPU reads from it run at well under 1 GB/s. HOST_COHERENT is required
    // too, because callers map/unmap without explicit invalidate/flush.
    const VkFlags cached = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                           | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    VkFlags mask = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const auto& memProty = mRuntime->mDevice->memProty();
    for (uint32_t i = 0; i < memProty.memoryTypeCount; ++i) {
        if ((memProty.memoryTypes[i].propertyFlags & cached) == cached) {
            mask = cached;
            break;
        }
    }
    std::shared_ptr<VulkanBuffer> res;
    res.reset(new VulkanBuffer(*mRuntime->mMemoryPool, false, size, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_CONCURRENT, mask));
    return res;
}

void VulkanBackend::copyGPUToGPUBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset) const {
    auto cmdbuffer = mCmdBufferForCopy;
    cmdbuffer->begin(0);
    VkBufferCopy bufferCopy;
    bufferCopy.size = size;
    bufferCopy.dstOffset = dstOffset;
    bufferCopy.srcOffset = srcOffset;
    vkCmdCopyBuffer(cmdbuffer->get(), srcBuffer, dstBuffer,
                    1, &bufferCopy);
    cmdbuffer->end();
    pushCommand(cmdbuffer->get());
    _finish();
}

void VulkanBackend::copyGPUToGPUBufferRegions(VkBuffer srcBuffer, VkBuffer dstBuffer, const VkBufferCopy* regions,
                                              uint32_t regionCount) const {
    if (regionCount == 0 || regions == nullptr) {
        return;
    }
    auto cmdbuffer = mCmdBufferForCopy;
    cmdbuffer->begin(0);
    vkCmdCopyBuffer(cmdbuffer->get(), srcBuffer, dstBuffer, regionCount, regions);
    cmdbuffer->end();
    pushCommand(cmdbuffer->get());
    _finish();
}

void VulkanBackend::copyToGPUBuffer(const void* src, VkBuffer buffer, VkDeviceSize size, VkDeviceSize offset) const {
    _requireHostBuffer(size);
    ::memcpy(mHostBuffer->map(), src, size);
    mHostBuffer->unmap();
    copyGPUToGPUBuffer(mHostBuffer->buffer(), buffer, size, 0, offset);
}
bool VulkanBackend::_copyWithHostImport(VkBuffer deviceBuffer, VkDeviceSize deviceOffset, void* host, size_t size,
                                        bool deviceToHost) const {
    // Below a few MB the staging memcpy is cheaper than pinning the pages.
    const size_t kMinImportBytes = 4 * 1024 * 1024;
    const auto& dev = *mRuntime->mDevice;
    if (!dev.supportHostImport() || size < kMinImportBytes || nullptr == host) {
        return false;
    }
    // Import whole pages around [host, host + size): the pointer and the size
    // must be multiples of minImportedHostPointerAlignment. Only the requested
    // range is ever written; the rest of the first / last page is untouched.
    const VkDeviceSize align = dev.hostImportAlignment();
    const uintptr_t start = reinterpret_cast<uintptr_t>(host);
    const uintptr_t base = start & ~static_cast<uintptr_t>(align - 1);
    const VkDeviceSize inner = start - base;
    const VkDeviceSize importSize = UP_DIV(inner + size, align) * align;
    void* basePtr = reinterpret_cast<void*>(base);

    VkMemoryHostPointerPropertiesEXT hostProps{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (VK_SUCCESS != dev.getMemoryHostPointerProperties(basePtr, &hostProps) || 0 == hostProps.memoryTypeBits) {
        return false;
    }
    VkExternalMemoryBufferCreateInfo externalInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.pNext       = &externalInfo;
    bufferInfo.size        = importSize;
    bufferInfo.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer hostBuffer = VK_NULL_HANDLE;
    if (VK_SUCCESS != vkCreateBuffer(dev.get(), &bufferInfo, nullptr, &hostBuffer)) {
        return false;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev.get(), hostBuffer, &req);
    // The CPU reads / writes the memory through its own pointer without any
    // map / invalidate, so only HOST_COHERENT types are acceptable.
    const uint32_t typeBits = req.memoryTypeBits & hostProps.memoryTypeBits;
    uint32_t typeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < dev.memProty().memoryTypeCount; ++i) {
        if ((typeBits & (1u << i))
            && (dev.memProty().memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            typeIndex = i;
            break;
        }
    }
    VkDeviceMemory hostMemory = VK_NULL_HANDLE;
    if (UINT32_MAX != typeIndex) {
        VkImportMemoryHostPointerInfoEXT importInfo{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
        importInfo.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        importInfo.pHostPointer = basePtr;
        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.pNext           = &importInfo;
        allocInfo.allocationSize  = importSize;
        allocInfo.memoryTypeIndex = typeIndex;
        if (VK_SUCCESS != vkAllocateMemory(dev.get(), &allocInfo, nullptr, &hostMemory)) {
            hostMemory = VK_NULL_HANDLE;
        }
    }
    if (VK_NULL_HANDLE == hostMemory || VK_SUCCESS != vkBindBufferMemory(dev.get(), hostBuffer, hostMemory, 0)) {
        if (VK_NULL_HANDLE != hostMemory) {
            vkFreeMemory(dev.get(), hostMemory, nullptr);
        }
        vkDestroyBuffer(dev.get(), hostBuffer, nullptr);
        return false;
    }

    VkBufferCopy region;
    region.size      = size;
    region.srcOffset = deviceToHost ? deviceOffset : inner;
    region.dstOffset = deviceToHost ? inner : deviceOffset;
    if (dev.transferCopy(deviceToHost ? deviceBuffer : hostBuffer, deviceToHost ? hostBuffer : deviceBuffer, region,
                         deviceToHost)) {
        vkDestroyBuffer(dev.get(), hostBuffer, nullptr);
        vkFreeMemory(dev.get(), hostMemory, nullptr);
        return true;
    }
    auto cmdbuffer = mCmdBufferForCopy;
    cmdbuffer->begin(0);
    if (deviceToHost) {
        vkCmdCopyBuffer(cmdbuffer->get(), deviceBuffer, hostBuffer, 1, &region);
        // Make the transfer writes available to host reads after the fence.
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmdbuffer->get(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                             &barrier, 0, nullptr, 0, nullptr);
    } else {
        vkCmdCopyBuffer(cmdbuffer->get(), hostBuffer, deviceBuffer, 1, &region);
    }
    cmdbuffer->end();
    pushCommand(cmdbuffer->get());
    _finish();
    vkDestroyBuffer(dev.get(), hostBuffer, nullptr);
    vkFreeMemory(dev.get(), hostMemory, nullptr);
    return true;
}

void VulkanBackend::_requireHostBuffer(size_t size) const {
    _finish();
    if (nullptr == mHostBuffer || mHostBuffer->size() < size) {
        mHostBuffer = createHostBuffer(size);
    }
}

void VulkanBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
#ifdef MNN_VULKAN_DEBUG
    AUTOTIME;
    MNN_PRINT("Src: ");
    for (int i=0; i<srcTensor->dimensions(); ++i) {
        MNN_PRINT("%d , ", srcTensor->length(i));
    }
    MNN_PRINT("\n");
    MNN_PRINT("Dst: ");
    for (int i=0; i<dstTensor->dimensions(); ++i) {
        MNN_PRINT("%d , ", dstTensor->length(i));
    }
    MNN_PRINT("\n");
#endif

    auto calculateCpSize = [this] (const Tensor* tensor) -> size_t {
            size_t eleSize = (size_t) tensor->elementSize();
            return (tensor->getType().code == halide_type_float && this->mUseFP16) ?
                (eleSize * sizeof(uint16_t)) :
                (eleSize * sizeof(float));
        };

    std::shared_ptr<Tensor> tempTensor;
    if (srcTensor->host<float>() != nullptr) {
        _finish();
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        auto buffer = reinterpret_cast<VulkanBuffer*>(dstTensor->deviceId());
        auto offset = TensorUtils::getDescribeOrigin(dstTensor)->offset;
        // host->gpu
        if(format != TensorUtils::getDescribe(srcTensor)->dimensionFormat) {
            tempTensor.reset(Tensor::create(dstTensor->shape(), dstTensor->getType(), nullptr, _convert(format)));
            MNNCPUCopyBuffer(srcTensor, tempTensor.get());
            srcTensor = tempTensor.get();
        }
        size_t cpSize = calculateCpSize(srcTensor);
        const bool float2half = srcTensor->getType().code == halide_type_float && mUseFP16;
        if (!float2half && _copyWithHostImport(buffer->buffer(), offset, srcTensor->host<void>(), cpSize, false)) {
            return;
        }
        _requireHostBuffer(cpSize);
        _copyTensorToBuffer(srcTensor, mHostBuffer.get(), 0, float2half);
        VkBufferCopy bufferCopy;
        bufferCopy.size = cpSize;
        bufferCopy.dstOffset = offset;
        bufferCopy.srcOffset = 0;
        if (!mRuntime->mDevice->transferCopy(mHostBuffer->buffer(), buffer->buffer(), bufferCopy, false)) {
            auto cmdbuffer = mCmdBufferForCopy;
            cmdbuffer->begin(0);
            vkCmdCopyBuffer(cmdbuffer->get(), mHostBuffer->buffer(), buffer->buffer(),
                            1, &bufferCopy);
            cmdbuffer->end();
            pushCommand(cmdbuffer->get());
            _finish();
        }
    } else if (dstTensor->host<float>() != nullptr) {
        // gpu->host
        _finish();
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        if (format != TensorUtils::getDescribe(srcTensor)->dimensionFormat) {
            tempTensor.reset(Tensor::create(srcTensor->shape(), dstTensor->getType(), nullptr, _convert(TensorUtils::getDescribe(srcTensor)->dimensionFormat)), [dstTensor](void* t) {
                Tensor* temp = (Tensor*)t;
                MNNCPUCopyBuffer(temp, dstTensor);
                delete temp;
            });
            dstTensor = tempTensor.get();
        }
        size_t cpSize = calculateCpSize(dstTensor);
        auto buffer = reinterpret_cast<VulkanBuffer*>(srcTensor->deviceId());
        auto offset = TensorUtils::getDescribeOrigin(srcTensor)->offset;
        const bool half2float = dstTensor->getType().code == halide_type_float && mUseFP16;
        if (!half2float && _copyWithHostImport(buffer->buffer(), offset, dstTensor->host<void>(), cpSize, true)) {
            return;  // tempTensor (if any) converts into the caller's tensor on release
        }
        _requireHostBuffer(cpSize);
        VkBufferCopy bufferCopy;
        bufferCopy.size = cpSize;
        bufferCopy.dstOffset = 0;
        bufferCopy.srcOffset = offset;
        if (!mRuntime->mDevice->transferCopy(buffer->buffer(), mHostBuffer->buffer(), bufferCopy, true)) {
            auto cmdbuffer = mCmdBufferForCopy;
            cmdbuffer->begin(0);
            vkCmdCopyBuffer(cmdbuffer->get(), buffer->buffer(), mHostBuffer->buffer(),
                            1, &bufferCopy);
            cmdbuffer->end();
            pushCommand(cmdbuffer->get());
            _finish();
        }
        _copyBufferToTensor(dstTensor, mHostBuffer.get(), 0, dstTensor->getType().code == halide_type_float && mUseFP16);
    } else if (srcTensor->deviceId() != 0 && dstTensor->deviceId() != 0) {
        // gpu->gpu
        auto format = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
        MNN_ASSERT(format == TensorUtils::getDescribe(srcTensor)->dimensionFormat);
        std::shared_ptr<VulkanCommandPool::Buffer> buffer( mRuntime->mCmdPool->allocBuffer());
        buffer->begin(0);
        VkBufferCopy bufferCopy;
        size_t cpSize = calculateCpSize(srcTensor);
        bufferCopy.size = cpSize;
        bufferCopy.dstOffset = TensorUtils::getDescribeOrigin(dstTensor)->offset;
        bufferCopy.srcOffset = TensorUtils::getDescribeOrigin(srcTensor)->offset;
        vkCmdCopyBuffer(buffer->get(), reinterpret_cast<VulkanBuffer*>(srcTensor->deviceId())->buffer(), reinterpret_cast<VulkanBuffer*>(dstTensor->deviceId())->buffer(),
                        1, &bufferCopy);
        buffer->end();
        pushCommand(buffer->get());
        _finish();
    }
}
int VulkanBackend::onSync(Tensor::MapType mtype, bool toCpu, const Tensor* dstTensor) {
    _finish();
    return 0;
}

bool VulkanBackend::addCreator(OpType t, Creator* c) {
    auto allKind = getCreatorMap();
    allKind->insert(std::make_pair(t, c));
    return true;
}

bool VulkanBackend::onGetTensorInfo(const Tensor* tensor, void* dstInfo) {
    if (nullptr == dstInfo) {
        return true;
    }
    auto vkBuffer = getBuffer(tensor);
    auto dst = (MNNVulkanTensorContent*)dstInfo;
    dst->buffer = std::get<0>(vkBuffer);
    dst->offset = std::get<2>(vkBuffer);
    dst->size = std::get<1>(vkBuffer);
    return true;
}

float VulkanBackend::getPipelineTime(const VulkanPipeline* pipeline, SharedPtr<VulkanLayout::DescriptorSet> des, std::vector<int> groupSize){
    std::shared_ptr<VulkanCommandPool::Buffer> cmd;
    cmd.reset(const_cast<VulkanCommandPool::Buffer *>(getPool().allocBuffer()));
    cmd->begin(0);
    mRuntime->mQueryPool->VulkanCmdResetQueryPool(cmd.get()->get());
    mRuntime->mQueryPool->VulkanCmdWriteTimestamp(cmd.get()->get(), 0);
    pipeline->bind(cmd.get()->get(), des->get());
    vkCmdDispatch(cmd.get()->get(), groupSize[0], groupSize[1], groupSize[2]);
    mRuntime->mQueryPool->VulkanCmdWriteTimestamp(cmd.get()->get(), 1);
    cmd->end();
    getPool().submitAndWait(cmd.get()->get());
    float time = mRuntime->mQueryPool->VulkanGetQueryPoolResults();
    return time;
}

std::vector<uint32_t> VulkanBackend::autoTunePipeline(const VulkanPipeline* pipeline, SharedPtr<VulkanLayout::DescriptorSet> des, std::vector<int> gws){
    std::vector<uint32_t> lws(3, 1);
    std::vector<int> groupSize(3, 1);
    std::vector<int> maxGroups(3, 1);
    int maxGroupSize = mRuntime->mDevice->getMaxComputeWorkGroupInvocations();
    mRuntime->mDevice->getMaxComputeWorkGroupSize(maxGroups);
    
    std::vector<uint32_t> lws_prefer(3, 1);
    float min_cost = -1.0f;
    
    while(lws[2] <= gws[2] && lws[2] <= maxGroups[2]) {
        lws[1] = 1;
        while(lws[1] <= gws[1] && lws[1] <= maxGroups[1]) {
            lws[0] = 1;
            while(lws[0] <= gws[0] && lws[0] <= maxGroups[0]) {
                if(lws[0]*lws[1]*lws[2] <= maxGroupSize) {
                    groupSize[0] = UP_DIV(gws[0], lws[0]);
                    groupSize[1] = UP_DIV(gws[1], lws[1]);
                    groupSize[2] = UP_DIV(gws[2], lws[2]);
                    
                    pipeline->changePipeline(lws);
                    auto cost_time = getPipelineTime(pipeline, des, groupSize);
                    if(cost_time < min_cost || min_cost < 0.0f) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                        lws_prefer[2] = lws[2];
                    }
                }
                lws[0]*=2;
            }
            lws[1]*=2;
        }
        lws[2]*=2;
    }
    
    pipeline->changePipeline(lws_prefer);
    return lws_prefer;
}

} // namespace MNN
