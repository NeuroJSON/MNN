// Standalone OpenCL test of conv_3d_buf_nc4dhw4 kernel on a stride-1 32->32 layer.
// If this works, MNN is the bug; if it fails, our kernel/buffer setup is the bug.

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
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    // ---- pick OpenCL device ----
    cl_uint nplat = 0;
    clGetPlatformIDs(0, nullptr, &nplat);
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);
    cl_platform_id plat = plats[0];
    cl_device_id dev;
    cl_uint ndev;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &ndev);
    char devname[256];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devname), devname, nullptr);
    fprintf(stderr, "[standalone] device: %s\n", devname);

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CHECK(err, "create context");
    /* Match MNN's queue: CL_QUEUE_PROFILING_ENABLE */
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
    CHECK(err, "create queue");

    // ---- build kernel ----
    std::string src = read_file("/home/users/fangq/space/git/Temp/MNN/source/backend/opencl/execution/cl/conv_3d_buf.cl");
    const char* src_str = src.c_str();
    size_t src_len = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_str, &src_len, &err);
    CHECK(err, "create program");

    const char* opts =
        "-DFLOAT=float -DFLOAT2=float2 -DFLOAT3=float3 -DFLOAT4=float4 "
        "-DCOMPUTE_FLOAT=float -DCOMPUTE_FLOAT4=float4 "
        "-DCONVERT_COMPUTE_FLOAT4=convert_float4 "
        "-DCONVERT_FLOAT4=convert_float4 ";

    err = clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::string log(log_sz, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, &log[0], nullptr);
        fprintf(stderr, "BUILD FAIL:\n%s\n", log.c_str());
        return 1;
    }
    cl_kernel k = clCreateKernel(prog, "conv_3d_buf_nc4dhw4", &err);
    CHECK(err, "create kernel");

    // ---- test the SAME shape as SIAM's failing stride-1 32->32 conv ----
    const int Cin = 32, Cout = 32;
    const int D = 192, H = 192, W = 128;
    const int kd = 3, kh = 3, kw = 3;
    const int sd = 1, sh = 1, sw = 1;
    const int pd = 1, ph = 1, pw = 1;
    const int Cinblk  = (Cin + 3) / 4;   // 8
    const int Coutblk = (Cout + 3) / 4;  // 8

    const size_t in_floats  = (size_t)Cinblk * D * H * W * 4;
    const size_t out_floats = (size_t)Coutblk * D * H * W * 4;
    const size_t w_floats   = (size_t)Coutblk * kd * kh * kw * Cinblk * 16;
    const size_t b_floats   = (size_t)Coutblk * 4;

    fprintf(stderr, "[standalone] in=%.1fMB out=%.1fMB weight=%zu bias=%zu\n",
            in_floats*4/1e6, out_floats*4/1e6, w_floats, b_floats);

    std::vector<float> h_in(in_floats, 1.0f);
    std::vector<float> h_w(w_floats, 0.01f);
    std::vector<float> h_b(b_floats, 0.5f);
    std::vector<float> h_out(out_floats, -7.0f);

    /* Replicate MNN's BufferPool flag exactly */
    const cl_mem_flags flag = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
    cl_mem in_buf  = clCreateBuffer(ctx, flag, in_floats * 4, nullptr, &err);  CHECK(err, "in buf");
    cl_mem w_buf   = clCreateBuffer(ctx, flag, w_floats  * 4, nullptr, &err);  CHECK(err, "w buf");
    cl_mem b_buf   = clCreateBuffer(ctx, flag, b_floats  * 4, nullptr, &err);  CHECK(err, "b buf");
    cl_mem out_buf = clCreateBuffer(ctx, flag, out_floats* 4, nullptr, &err);  CHECK(err, "out buf");

    clEnqueueWriteBuffer(queue, in_buf,  CL_TRUE, 0, in_floats *4, h_in.data(),  0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, w_buf,   CL_TRUE, 0, w_floats  *4, h_w.data(),   0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, b_buf,   CL_TRUE, 0, b_floats  *4, h_b.data(),   0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, out_buf, CL_TRUE, 0, out_floats*4, h_out.data(), 0, nullptr, nullptr);

    cl_uint gsd0 = W, gsd1 = D * H, gsd2 = Coutblk;
    int idx = 0;
    CHECK(clSetKernelArg(k, idx++, sizeof(cl_uint), &gsd0), "arg gsd0");
    CHECK(clSetKernelArg(k, idx++, sizeof(cl_uint), &gsd1), "arg gsd1");
    CHECK(clSetKernelArg(k, idx++, sizeof(cl_uint), &gsd2), "arg gsd2");
    CHECK(clSetKernelArg(k, idx++, sizeof(cl_mem), &in_buf),  "arg in");
    CHECK(clSetKernelArg(k, idx++, sizeof(cl_mem), &w_buf),   "arg w");
    CHECK(clSetKernelArg(k, idx++, sizeof(cl_mem), &b_buf),   "arg b");
    CHECK(clSetKernelArg(k, idx++, sizeof(cl_mem), &out_buf), "arg out");
    int vCinblk = Cinblk, vD = D, vH = H, vW = W;
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vCinblk), "arg cinblk");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vD), "in_d");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vH), "in_h");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vW), "in_w");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vD), "out_d");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vH), "out_h");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vW), "out_w");
    int vKd = kd, vKh = kh, vKw = kw, vSd = sd, vSh = sh, vSw = sw, vPd = pd, vPh = ph, vPw = pw;
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vKd), "kd");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vKh), "kh");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vKw), "kw");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vSd), "sd");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vSh), "sh");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vSw), "sw");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vPd), "pd");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vPh), "ph");
    CHECK(clSetKernelArg(k, idx++, sizeof(int), &vPw), "pw");

    size_t gws[3] = { (size_t)W, (size_t)(D * H), (size_t)Coutblk };
    size_t lws[3] = { 8, 32, 1 };

    /* Round up GWS */
    for (int i = 0; i < 3; ++i) {
        gws[i] = ((gws[i] + lws[i] - 1) / lws[i]) * lws[i];
    }
    fprintf(stderr, "[standalone] launching gws=(%zu,%zu,%zu) lws=(%zu,%zu,%zu)\n",
            gws[0], gws[1], gws[2], lws[0], lws[1], lws[2]);

    /* Enqueue the kernel 76 times sequentially (matching SIAM per-tile count) */
    for (int rep = 0; rep < 76; ++rep) {
        cl_event ev;
        err = clEnqueueNDRangeKernel(queue, k, 3, nullptr, gws, lws, 0, nullptr, &ev);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[standalone] rep %d enqueue err=%d\n", rep, err);
        }
        err = clWaitForEvents(1, &ev);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[standalone] rep %d wait err=%d\n", rep, err);
        }
        clReleaseEvent(ev);
    }
    fprintf(stderr, "[standalone] all 76 launches done\n");

    /* Read back output */
    clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, out_floats*4, h_out.data(), 0, nullptr, nullptr);

    /* Check: how many of the output floats are still the sentinel -7.0? */
    size_t still_unchanged = 0, modified = 0;
    float sum = 0;
    for (size_t i = 0; i < out_floats; ++i) {
        if (h_out[i] == -7.0f) still_unchanged++;
        else { modified++; sum += h_out[i]; }
    }
    fprintf(stderr, "[standalone] output: %zu still=-7.0 (unchanged), %zu modified (sum=%g)\n",
            still_unchanged, modified, sum);
    fprintf(stderr, "[standalone] first 8 output values: ");
    for (int i = 0; i < 8; ++i) fprintf(stderr, "%g ", h_out[i]);
    fprintf(stderr, "\n");
    return 0;
}
