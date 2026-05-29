// Minimal MNN runner -- load /tmp/tiny_conv3d.mnn and run inference.
// Reports which Conv3D ops fired (via kernel printf hooked in conv_3d_buf.cl).
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1] : "/tmp/tiny_conv3d.mnn";
    fprintf(stderr, "[runner] loading %s\n", model_path);

    auto* interp = MNN::Interpreter::createFromFile(model_path);
    if (!interp) { fprintf(stderr, "load failed\n"); return 1; }

    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_OPENCL;
    cfg.numThread = 4 | 64;  /* TUNING_WIDE | MEMORY_BUFFER */
    MNN::BackendConfig bcfg;
    bcfg.precision = MNN::BackendConfig::Precision_High;  /* fp32 */
    cfg.backendConfig = &bcfg;

    auto* session = interp->createSession(cfg);
    if (!session) { fprintf(stderr, "createSession failed\n"); return 1; }

    auto* in = interp->getSessionInput(session, nullptr);
    fprintf(stderr, "[runner] input dims: %d (", in->dimensions());
    for (int i = 0; i < in->dimensions(); ++i) fprintf(stderr, "%d,", in->length(i));
    fprintf(stderr, ")\n");

    /* Fill input with synthetic values */
    std::vector<float> hin(in->elementSize(), 0.5f);
    auto h_tensor = MNN::Tensor::create<float>(
        std::vector<int>(in->shape().begin(), in->shape().end()),
        hin.data(), MNN::Tensor::CAFFE);
    in->copyFromHostTensor(h_tensor);
    delete h_tensor;

    fprintf(stderr, "[runner] running session...\n");
    auto code = interp->runSession(session);
    fprintf(stderr, "[runner] runSession returned %d\n", (int)code);

    /* Read output */
    auto* out = interp->getSessionOutput(session, nullptr);
    auto h_out = MNN::Tensor::create<float>(
        std::vector<int>(out->shape().begin(), out->shape().end()),
        nullptr, MNN::Tensor::CAFFE);
    out->copyToHostTensor(h_out);

    /* Summarize output */
    float* p = h_out->host<float>();
    int n = h_out->elementSize();
    int nz = 0;
    double s = 0;
    for (int i = 0; i < n; ++i) {
        if (p[i] != 0.0f) nz++;
        s += p[i];
    }
    fprintf(stderr, "[runner] output: %d elements, %d non-zero, sum=%g, first 8: ",
            n, nz, s);
    for (int i = 0; i < 8 && i < n; ++i) fprintf(stderr, "%g ", p[i]);
    fprintf(stderr, "\n");
    delete h_out;

    interp->releaseSession(session);
    MNN::Interpreter::destroy(interp);
    return 0;
}
