/*
 * Native 3D transposed convolution kernel for MNN's OpenCL buffer
 * backend.
 *
 * Replaces the GeometryConvTranspose3D fallback that decomposes
 * ConvTranspose3D into MatMul + Col2Im + Reduce + Bias-Add + Activate
 * via a virtual (1, batch*oc*kd*kh*kw, batch*oc*od*oh*ow) tensor.
 * At SIAM v0.3 decoder scale (oc=256, k=3, od*oh*ow ~= 9216) the
 * intermediate Reduce kernel either silently zeros or kernel-faults
 * on NVIDIA OpenCL, blocking the 5 post-Concat decoder Conv3D ops.
 *
 * Algorithm: GATHER formulation.
 *
 *   For each output voxel (od, oh, ow) and each 4-output-channel
 *   block, accumulate contributions from input voxels that map onto
 *   this output position through the ConvTranspose3D math:
 *
 *       od = id * sd - pd + kd     =>   id = (od + pd - kd) / sd
 *
 *   `id` is valid only when (od + pd - kd) is divisible by sd AND
 *   0 <= id < in_d. Same for h and w axes. Voxels added by the ONNX
 *   output_padding extension naturally produce bias-only outputs
 *   because no (id, kd) pair maps to them.
 *
 * Data layout: identical to conv_3d_buf -- input / output use MNN's
 * NC4DHW4 packed layout [Cblock, N, D, H, W, 4]; weights pack into
 * [Cout_block, Kd, Kh, Kw, Cin_block, 4_cin, 4_cout] (the host
 * Deconv3DBufExecution.cpp transforms ONNX's (Cin, Cout, Kd, Kh, Kw)
 * into this kernel-friendly layout at session-init time).
 *
 * SIAM ConvTranspose3D shape set covered by this kernel:
 *
 *     kernel 3x3x3   stride (2,2,2)   pad (1,1,1)   output_pad (1,1,1)
 *
 * (6 ops total; channel pairs 320->320 x2, 320->256, 256->128,
 * 128->64, 64->32.)
 *
 * Group=1, dilation=1 only (SIAM does not exercise other variants).
 */

#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, \
    __private const int global_size_dim1, \
    __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(d0, d1, d2)                            \
    if (d0 >= global_size_dim0 ||                                    \
        d1 >= global_size_dim1 ||                                    \
        d2 >= global_size_dim2) {                                    \
        return;                                                      \
    }

/*
 * deconv_3d_buf_nc4dhw4
 *
 * Global work size:
 *   dim0 : out_w
 *   dim1 : out_d * out_h
 *   dim2 : out_cblock           (= ceil(Cout / 4))   -- batch=1 baked in
 *
 * Each work-item produces one float4 of output (4 consecutive output
 * channels at one DHW voxel).
 */
__kernel void deconv_3d_buf_nc4dhw4(GLOBAL_SIZE_3_DIMS
                                    __global const FLOAT *input,
                                    __global const FLOAT *weight,
                                    __global const FLOAT *bias,
                                    __global FLOAT *output,
                                    __private const int in_cblock,
                                    __private const int in_d,
                                    __private const int in_h,
                                    __private const int in_w,
                                    __private const int out_d,
                                    __private const int out_h,
                                    __private const int out_w,
                                    __private const int kd,
                                    __private const int kh,
                                    __private const int kw,
                                    __private const int sd,
                                    __private const int sh,
                                    __private const int sw,
                                    __private const int pd,
                                    __private const int ph,
                                    __private const int pw) {
    const int gid_w  = get_global_id(0);
    const int gid_dh = get_global_id(1);
    const int gid_cb = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(gid_w, gid_dh, gid_cb);

    const int ox     = gid_w;
    const int oh     = gid_dh % out_h;
    const int od     = gid_dh / out_h;
    const int oc_blk = gid_cb;

    COMPUTE_FLOAT4 acc = CONVERT_COMPUTE_FLOAT4(vload4(oc_blk, bias));

    /* Weight base offset for this oc_blk -- same packing as conv_3d_buf:
     *   [oc_blk, kd, kh, kw, ic_blk, 4_cin, 4_cout]   16 floats per tile. */
    const int w_kvol    = kd * kh * kw;
    const int w_per_oc  = w_kvol * in_cblock * 4 * 4;
    const int w_oc_base = oc_blk * w_per_oc;

    for (int kdi = 0; kdi < kd; ++kdi) {
        const int idz_num = od + pd - kdi;

        if (idz_num < 0) {
            continue;
        }

        if ((idz_num % sd) != 0) {
            continue;
        }

        const int idz = idz_num / sd;

        if (idz >= in_d) {
            continue;
        }

        for (int khi = 0; khi < kh; ++khi) {
            const int idy_num = oh + ph - khi;

            if (idy_num < 0) {
                continue;
            }

            if ((idy_num % sh) != 0) {
                continue;
            }

            const int idy = idy_num / sh;

            if (idy >= in_h) {
                continue;
            }

            for (int kwi = 0; kwi < kw; ++kwi) {
                const int idx_num = ox + pw - kwi;

                if (idx_num < 0) {
                    continue;
                }

                if ((idx_num % sw) != 0) {
                    continue;
                }

                const int idx = idx_num / sw;

                if (idx >= in_w) {
                    continue;
                }

                const int w_kpos_base = w_oc_base
                                        + ((kdi * kh + khi) * kw + kwi) * in_cblock * 16;

                for (int ic_blk = 0; ic_blk < in_cblock; ++ic_blk) {
                    /* Input layout: [cblock, batch=1, D, H, W, 4] */
                    const int in_voxel = ((ic_blk * in_d + idz) * in_h + idy) * in_w + idx;
                    COMPUTE_FLOAT4 in_v = CONVERT_COMPUTE_FLOAT4(vload4(in_voxel, input));

                    const int w_offset = w_kpos_base + ic_blk * 16;
                    COMPUTE_FLOAT4 w0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_offset));
                    COMPUTE_FLOAT4 w1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_offset +  4));
                    COMPUTE_FLOAT4 w2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_offset +  8));
                    COMPUTE_FLOAT4 w3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_offset + 12));

                    acc = mad((COMPUTE_FLOAT4)(in_v.x), w0, acc);
                    acc = mad((COMPUTE_FLOAT4)(in_v.y), w1, acc);
                    acc = mad((COMPUTE_FLOAT4)(in_v.z), w2, acc);
                    acc = mad((COMPUTE_FLOAT4)(in_v.w), w3, acc);
                }
            }
        }
    }

    const int out_voxel = ((oc_blk * out_d + od) * out_h + oh) * out_w + ox;
    vstore4(CONVERT_FLOAT4(acc), out_voxel, output);
}

/*
 * deconv_3d_buf_nchw
 *
 * Variant for tensors that flow through a ConvertTensor (NC4HW4 -> NCHW)
 * before the ConvTranspose3D op. The actual buffer contains 5D NCDHW
 * data:  input[c, d, h, w] = input[c*D*H*W + d*H*W + h*W + w].
 *
 * SIAM's decoder uses this path: every ConvTranspose3D in the network
 * has a preceding ConvertTensor that materializes its input into NCHW
 * layout (so that the downstream Concat can see matching layouts from
 * both the deconv branch and the skip branch). Output stays NCHW too
 * for the Concat to consume directly.
 *
 * Each work-item still produces 4 output channels at one DHW voxel,
 * but reads / writes are scalar (strided) loads / stores instead of
 * vload4 / vstore4. Weights stay in the same packed kernel-friendly
 * layout the host builds.
 */
__kernel void deconv_3d_buf_nchw(GLOBAL_SIZE_3_DIMS
                                 __global const FLOAT *input,
                                 __global const FLOAT *weight,
                                 __global const FLOAT *bias,
                                 __global FLOAT *output,
                                 __private const int in_c,
                                 __private const int in_d,
                                 __private const int in_h,
                                 __private const int in_w,
                                 __private const int out_c,
                                 __private const int out_d,
                                 __private const int out_h,
                                 __private const int out_w,
                                 __private const int kd,
                                 __private const int kh,
                                 __private const int kw,
                                 __private const int sd,
                                 __private const int sh,
                                 __private const int sw,
                                 __private const int pd,
                                 __private const int ph,
                                 __private const int pw) {
    const int gid_w  = get_global_id(0);
    const int gid_dh = get_global_id(1);
    const int gid_cb = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(gid_w, gid_dh, gid_cb);

    const int ox     = gid_w;
    const int oh     = gid_dh % out_h;
    const int od     = gid_dh / out_h;
    const int oc_blk = gid_cb;
    const int oc0    = oc_blk * 4;

    COMPUTE_FLOAT4 acc = CONVERT_COMPUTE_FLOAT4(vload4(oc_blk, bias));

    const int in_cblock  = (in_c + 3) >> 2;
    const int in_spatial = in_d * in_h * in_w;
    const int w_kvol     = kd * kh * kw;
    const int w_per_oc   = w_kvol * in_cblock * 4 * 4;
    const int w_oc_base  = oc_blk * w_per_oc;

    for (int kdi = 0; kdi < kd; ++kdi) {
        const int idz_num = od + pd - kdi;
        if (idz_num < 0) continue;
        if ((idz_num % sd) != 0) continue;
        const int idz = idz_num / sd;
        if (idz >= in_d) continue;

        for (int khi = 0; khi < kh; ++khi) {
            const int idy_num = oh + ph - khi;
            if (idy_num < 0) continue;
            if ((idy_num % sh) != 0) continue;
            const int idy = idy_num / sh;
            if (idy >= in_h) continue;

            for (int kwi = 0; kwi < kw; ++kwi) {
                const int idx_num = ox + pw - kwi;
                if (idx_num < 0) continue;
                if ((idx_num % sw) != 0) continue;
                const int idx = idx_num / sw;
                if (idx >= in_w) continue;

                const int w_kpos_base = w_oc_base
                                        + ((kdi * kh + khi) * kw + kwi) * in_cblock * 16;
                const int spat_off = (idz * in_h + idy) * in_w + idx;

                for (int ic_blk = 0; ic_blk < in_cblock; ++ic_blk) {
                    const int c_base = ic_blk * 4;

                    /* Read 4 input channels (strided by in_spatial), zero-pad
                     * channels past in_c so the inner loop runs to completion
                     * without boundary branches. */
                    COMPUTE_FLOAT in_x = (c_base + 0 < in_c) ? CONVERT_COMPUTE_FLOAT(input[(c_base + 0) * in_spatial + spat_off]) : (COMPUTE_FLOAT)0.0f;
                    COMPUTE_FLOAT in_y = (c_base + 1 < in_c) ? CONVERT_COMPUTE_FLOAT(input[(c_base + 1) * in_spatial + spat_off]) : (COMPUTE_FLOAT)0.0f;
                    COMPUTE_FLOAT in_z = (c_base + 2 < in_c) ? CONVERT_COMPUTE_FLOAT(input[(c_base + 2) * in_spatial + spat_off]) : (COMPUTE_FLOAT)0.0f;
                    COMPUTE_FLOAT in_w_= (c_base + 3 < in_c) ? CONVERT_COMPUTE_FLOAT(input[(c_base + 3) * in_spatial + spat_off]) : (COMPUTE_FLOAT)0.0f;

                    const int w_offset = w_kpos_base + ic_blk * 16;
                    COMPUTE_FLOAT4 w0 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_offset));
                    COMPUTE_FLOAT4 w1 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_offset +  4));
                    COMPUTE_FLOAT4 w2 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_offset +  8));
                    COMPUTE_FLOAT4 w3 = CONVERT_COMPUTE_FLOAT4(vload4(0, weight + w_offset + 12));

                    acc = mad((COMPUTE_FLOAT4)(in_x), w0, acc);
                    acc = mad((COMPUTE_FLOAT4)(in_y), w1, acc);
                    acc = mad((COMPUTE_FLOAT4)(in_z), w2, acc);
                    acc = mad((COMPUTE_FLOAT4)(in_w_), w3, acc);
                }
            }
        }
    }

    /* NCHW write: scalar stores at strided channel offsets. */
    const int out_spatial = out_d * out_h * out_w;
    const int out_spat_off = (od * out_h + oh) * out_w + ox;
    if (oc0 + 0 < out_c) output[(oc0 + 0) * out_spatial + out_spat_off] = CONVERT_FLOAT(acc.x);
    if (oc0 + 1 < out_c) output[(oc0 + 1) * out_spatial + out_spat_off] = CONVERT_FLOAT(acc.y);
    if (oc0 + 2 < out_c) output[(oc0 + 2) * out_spatial + out_spat_off] = CONVERT_FLOAT(acc.z);
    if (oc0 + 3 < out_c) output[(oc0 + 3) * out_spatial + out_spat_off] = CONVERT_FLOAT(acc.w);
}
