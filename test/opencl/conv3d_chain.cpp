// Standalone reproducer of the MNN+NVIDIA Conv3D failure pattern.
// Chains Conv3D + fake-LayerNorm + fake-ReLU repeatedly, with buffer
// reuse mimicking MNN's BufferPool. If Conv3D launches stop firing
// the kernel-side printf after a few iterations, we've reproduced
// the MNN pattern outside MNN -> root cause is NVIDIA driver
// behavior under interleaved-kernel patterns, and the fix is
// clFinish() between native Conv3D ops in MNN.

#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>

#define CHECK(err, what) do { \
    if ((err) != CL_SUCCESS) { fprintf(stderr, "FAIL %s: %d\n", (what), (int)(err)); return 1; } \
} while(0)

static std::string read_file(const char* p) {
    std::ifstream f(p);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

/* Fake LayerNorm: reads input[i], writes (input[i] - 0.5) * 2.0 to output[i].
 * Pure pointwise, just to mimic MNN running ANOTHER kernel on the
 * intermediate buffer between Conv3Ds. */
static const char* g_aux_src = R"(
#define GLOBAL_SIZE_1 __private const int global_size
#define DEAL(d) if ((d) >= global_size) { return; }

__kernel void fake_layernorm(GLOBAL_SIZE_1,
                             __global const FLOAT *input,
                             __global FLOAT *output)
{
    int i = get_global_id(0);
    DEAL(i);
    float v = input[i];
    output[i] = (v - 0.5f) * 2.0f;
}

__kernel void fake_relu(GLOBAL_SIZE_1,
                        __global const FLOAT *input,
                        __global FLOAT *output)
{
    int i = get_global_id(0);
    DEAL(i);
    float v = input[i];
    output[i] = v > 0.0f ? v : 0.0f;
}
)";

int main(int argc, char** argv) {
    cl_uint nplat; clGetPlatformIDs(0, nullptr, &nplat);
    std::vector<cl_platform_id> plats(nplat); clGetPlatformIDs(nplat, plats.data(), nullptr);
    cl_device_id dev; cl_uint ndev;
    clGetDeviceIDs(plats[0], CL_DEVICE_TYPE_GPU, 1, &dev, &ndev);
    char devname[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devname), devname, nullptr);
    fprintf(stderr, "[chain] device: %s\n", devname);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CHECK(err, "ctx");
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev, props, &err); CHECK(err, "queue");

    /* --- build conv3d kernel --- */
    std::string conv_src = read_file("/home/users/fangq/space/git/Temp/MNN/source/backend/opencl/execution/cl/conv_3d_buf.cl");
    const char* conv_str = conv_src.c_str(); size_t conv_len = conv_src.size();
    cl_program conv_prog = clCreateProgramWithSource(ctx, 1, &conv_str, &conv_len, &err); CHECK(err, "conv prog");
    const char* opts =
        "-DFLOAT=float -DFLOAT4=float4 "
        "-DCOMPUTE_FLOAT=float -DCOMPUTE_FLOAT4=float4 "
        "-DCONVERT_COMPUTE_FLOAT4=convert_float4 -DCONVERT_FLOAT4=convert_float4 ";
    err = clBuildProgram(conv_prog, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz; clGetProgramBuildInfo(conv_prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::string log(log_sz, '\0');
        clGetProgramBuildInfo(conv_prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, &log[0], nullptr);
        fprintf(stderr, "conv BUILD FAIL:\n%s\n", log.c_str()); return 1;
    }
    cl_kernel conv_k = clCreateKernel(conv_prog, "conv_3d_buf_nc4dhw4", &err); CHECK(err, "conv kernel");

    /* --- build aux (fake LN + ReLU) --- */
    size_t aux_len = strlen(g_aux_src);
    cl_program aux_prog = clCreateProgramWithSource(ctx, 1, &g_aux_src, &aux_len, &err); CHECK(err, "aux prog");
    err = clBuildProgram(aux_prog, 1, &dev, opts, nullptr, nullptr); CHECK(err, "aux build");
    cl_kernel ln_k   = clCreateKernel(aux_prog, "fake_layernorm", &err); CHECK(err, "ln kernel");
    cl_kernel relu_k = clCreateKernel(aux_prog, "fake_relu", &err);      CHECK(err, "relu kernel");

    /* --- conv3d shape (matches SIAM's failing stride-1 32->32 layer) --- */
    const int Cin = 32, Cout = 32;
    const int D = 192, H = 192, W = 128;
    const int kd = 3, kh = 3, kw = 3;
    const int sd = 1, sh = 1, sw = 1;
    const int pd = 1, ph = 1, pw = 1;
    const int Cinblk  = (Cin + 3) / 4;
    const int Coutblk = (Cout + 3) / 4;
    const size_t fea_floats = (size_t)Coutblk * D * H * W * 4;
    const size_t w_floats   = (size_t)Coutblk * kd * kh * kw * Cinblk * 16;
    const size_t b_floats   = (size_t)Coutblk * 4;
    fprintf(stderr, "[chain] feature volume = %.1fMB per buffer\n", fea_floats*4/1e6);

    /* --- allocate buffers like MNN: CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR --- */
    const cl_mem_flags flag = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
    cl_mem w_buf   = clCreateBuffer(ctx, flag, w_floats *4, nullptr, &err); CHECK(err, "w");
    cl_mem b_buf   = clCreateBuffer(ctx, flag, b_floats *4, nullptr, &err); CHECK(err, "b");
    /* 8 feature buffers, rotated like MNN's pool */
    const int NFEA = 8;
    std::vector<cl_mem> feas(NFEA);
    for (int i = 0; i < NFEA; ++i) {
        feas[i] = clCreateBuffer(ctx, flag, fea_floats*4, nullptr, &err); CHECK(err, "fea_i");
    }
    cl_mem fea_a = feas[0]; cl_mem fea_b = feas[1]; /* kept for compatibility below */

    std::vector<float> seed(fea_floats, 1.0f);
    std::vector<float> wseed(w_floats, 0.01f);
    std::vector<float> bseed(b_floats, 0.0f);
    clEnqueueWriteBuffer(queue, w_buf, CL_TRUE, 0, w_floats *4, wseed.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, b_buf, CL_TRUE, 0, b_floats *4, bseed.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, fea_a, CL_TRUE, 0, fea_floats*4, seed.data(),  0, nullptr, nullptr);

    /* --- conv3d kernel arg binder --- */
    auto bind_conv = [&](cl_mem in_buf, cl_mem out_buf) {
        cl_uint gsd0 = W, gsd1 = D*H, gsd2 = Coutblk;
        int idx = 0;
        clSetKernelArg(conv_k, idx++, sizeof(cl_uint), &gsd0);
        clSetKernelArg(conv_k, idx++, sizeof(cl_uint), &gsd1);
        clSetKernelArg(conv_k, idx++, sizeof(cl_uint), &gsd2);
        clSetKernelArg(conv_k, idx++, sizeof(cl_mem), &in_buf);
        clSetKernelArg(conv_k, idx++, sizeof(cl_mem), &w_buf);
        clSetKernelArg(conv_k, idx++, sizeof(cl_mem), &b_buf);
        clSetKernelArg(conv_k, idx++, sizeof(cl_mem), &out_buf);
        int vCinblk = Cinblk, vD=D, vH=H, vW=W, vKd=kd, vKh=kh, vKw=kw, vSd=sd, vSh=sh, vSw=sw, vPd=pd, vPh=ph, vPw=pw;
        clSetKernelArg(conv_k, idx++, sizeof(int), &vCinblk);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vD);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vH);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vW);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vD);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vH);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vW);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vKd);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vKh);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vKw);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vSd);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vSh);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vSw);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vPd);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vPh);
        clSetKernelArg(conv_k, idx++, sizeof(int), &vPw);
    };

    auto bind_aux = [&](cl_kernel k, cl_mem in_buf, cl_mem out_buf, cl_uint n) {
        int idx = 0;
        clSetKernelArg(k, idx++, sizeof(cl_uint), &n);
        clSetKernelArg(k, idx++, sizeof(cl_mem), &in_buf);
        clSetKernelArg(k, idx++, sizeof(cl_mem), &out_buf);
    };

    size_t conv_gws[3] = { (size_t)W, (size_t)(D*H), (size_t)Coutblk };
    size_t conv_lws[3] = { 8, 32, 1 };
    for (int i = 0; i < 3; ++i)
        conv_gws[i] = ((conv_gws[i] + conv_lws[i] - 1) / conv_lws[i]) * conv_lws[i];

    size_t aux_gws[1] = { fea_floats };
    size_t aux_lws[1] = { 256 };
    aux_gws[0] = ((aux_gws[0] + aux_lws[0] - 1) / aux_lws[0]) * aux_lws[0];
    cl_uint aux_n = (cl_uint)fea_floats;

    /* --- chain: 30 iterations of (conv3d -> LN -> Scale -> ReLU -> ConvertTensor -> ConvertTensor),
     * rotating through 8 buffers like MNN's pool. */
    int conv_fails = 0, conv_ok = 0;
    for (int iter = 0; iter < 30; ++iter) {
        cl_mem cin  = feas[(iter * 3 + 0) % NFEA];
        cl_mem cout = feas[(iter * 3 + 1) % NFEA];
        cl_mem aux1 = feas[(iter * 3 + 2) % NFEA];
        cl_mem aux2 = feas[(iter * 3 + 3) % NFEA];

        /* Conv3D: cin -> cout */
        bind_conv(cin, cout);
        cl_event ev;
        err = clEnqueueNDRangeKernel(queue, conv_k, 3, nullptr, conv_gws, conv_lws, 0, nullptr, &ev);
        if (err != CL_SUCCESS) { fprintf(stderr, "[chain] iter %d conv enqueue err=%d\n", iter, err); conv_fails++; continue; }
        err = clWaitForEvents(1, &ev);
        if (err != CL_SUCCESS) { fprintf(stderr, "[chain] iter %d conv wait err=%d\n", iter, err); conv_fails++; }
        else conv_ok++;
        clReleaseEvent(ev);

        /* fake LN: cout -> aux1 */
        bind_aux(ln_k, cout, aux1, aux_n);
        clEnqueueNDRangeKernel(queue, ln_k, 1, nullptr, aux_gws, aux_lws, 0, nullptr, nullptr);

        /* fake Scale (reuse LN kernel): aux1 -> aux2 */
        bind_aux(ln_k, aux1, aux2, aux_n);
        clEnqueueNDRangeKernel(queue, ln_k, 1, nullptr, aux_gws, aux_lws, 0, nullptr, nullptr);

        /* fake ReLU: aux2 -> cin (overwrites old input -- mimics MNN buffer reuse) */
        bind_aux(relu_k, aux2, cin, aux_n);
        clEnqueueNDRangeKernel(queue, relu_k, 1, nullptr, aux_gws, aux_lws, 0, nullptr, nullptr);

        /* fake ConvertTensor (copy buffer): cin -> aux1 */
        clEnqueueCopyBuffer(queue, cin, aux1, 0, 0, fea_floats * 4, 0, nullptr, nullptr);

        /* fake ConvertTensor #2: aux1 -> aux2 */
        clEnqueueCopyBuffer(queue, aux1, aux2, 0, 0, fea_floats * 4, 0, nullptr, nullptr);

        if (iter % 5 == 0) fprintf(stderr, "[chain] iter %d done\n", iter);
    }
    fprintf(stderr, "[chain] conv ok=%d fail=%d\n", conv_ok, conv_fails);

    /* Read back fea_a/fea_b first slot to confirm we got SOMETHING */
    std::vector<float> probe(8);
    clEnqueueReadBuffer(queue, fea_a, CL_TRUE, 0, 32, probe.data(), 0, nullptr, nullptr);
    fprintf(stderr, "[chain] fea_a[0..7]: ");
    for (int i = 0; i < 8; ++i) fprintf(stderr, "%g ", probe[i]);
    fprintf(stderr, "\n");
    clEnqueueReadBuffer(queue, fea_b, CL_TRUE, 0, 32, probe.data(), 0, nullptr, nullptr);
    fprintf(stderr, "[chain] fea_b[0..7]: ");
    for (int i = 0; i < 8; ++i) fprintf(stderr, "%g ", probe[i]);
    fprintf(stderr, "\n");
    return 0;
}
