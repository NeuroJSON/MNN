/*
 * Native 3D convolution kernel for MNN's OpenCL buffer backend.
 *
 * Replaces the GeometryConv3D fallback that decomposes Conv3D into
 * ~40 Conv2D + Im2Col + Unpack + Transpose ops per layer, producing
 * the ~3300-kernel-launch fan-out that makes SIAM-class 3D U-Nets
 * kernel-launch-bound on real GPUs.
 *
 * Data layout
 * -----------
 * Input / output use MNN's standard OpenCL buffer layout for 5D
 * tensors: `[Cblock, N, D, H, W, 4]` where Cblock = ceil(C / 4) and
 * the trailing 4 is the channel-pack. For batch=1 (always true for
 * sliding-window inference):
 *
 *     offset_voxel = (cblock * D + d) * H + h) * W + w     // float4
 *     pixel = vload4(offset_voxel, base)
 *
 * Weights are repacked at session creation into a kernel-friendly
 * layout that the host's Conv3DBufExecution writes:
 *
 *     [Cout_block, Kd, Kh, Kw, Cin_block, 4_cin, 4_cout]   // 16 floats
 *
 * For each (cout_block, kd, kh, kw, cin_block) tile, the kernel
 * loads 4 input channels via vload4, dot-products them against a
 * 4x4 weight slab, and accumulates into a 4-channel output vector.
 *
 * Bias layout is plain [Cout_block * 4] floats; we vload4 a
 * Cout-aligned chunk per work-item.
 *
 * SIAM Conv3D shape set covered by this single kernel:
 *
 *     kernel 1x1x1   stride (1,1,1) pad (0,0,0)
 *     kernel 3x3x3   stride (1,1,1) pad (1,1,1)
 *     kernel 3x3x3   stride (2,2,2) pad (1,1,1)
 *     kernel 3x3x3   stride (2,2,1) pad (1,1,1)
 *
 * All group=1, dilation=1. No relu/relu6 fusion (SIAM uses
 * InstanceNorm + LeakyReLU as separate ops between Conv3D blocks).
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
 * conv_3d_buf_nc4dhw4
 *
 * Global work size:
 *   dim0 : out_w
 *   dim1 : out_d * out_h
 *   dim2 : out_cblock        (= ceil(Cout / 4))   -- batch=1 baked in
 *
 * Each work-item produces one float4 of output (4 consecutive output
 * channels at one DHW voxel).
 */
__kernel void conv_3d_buf_nc4dhw4(GLOBAL_SIZE_3_DIMS
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

    /* Accumulator: 4 consecutive output channels at (od, oh, ox). */
    COMPUTE_FLOAT4 acc = CONVERT_COMPUTE_FLOAT4(vload4(oc_blk, bias));

    /* Receptive-field corner in the input volume. */
    const int in_d_base = od * sd - pd;
    const int in_h_base = oh * sh - ph;
    const int in_w_base = ox * sw - pw;

    /* Weight base offset for this oc_blk:
     *   [oc_blk, kd, kh, kw, ic_blk, 4_cin, 4_cout]
     * Per (kd, kh, kw, ic_blk) tile = 16 floats = 4 float4. */
    const int w_kvol   = kd * kh * kw;
    const int w_per_oc = w_kvol * in_cblock * 4 * 4;
    const int w_oc_base = oc_blk * w_per_oc;

    for (int kdi = 0; kdi < kd; ++kdi) {
        const int idz = in_d_base + kdi;

        if (idz < 0 || idz >= in_d) {
            continue;
        }

        for (int khi = 0; khi < kh; ++khi) {
            const int idy = in_h_base + khi;

            if (idy < 0 || idy >= in_h) {
                continue;
            }

            for (int kwi = 0; kwi < kw; ++kwi) {
                const int idx = in_w_base + kwi;

                if (idx < 0 || idx >= in_w) {
                    continue;
                }

                /* Weight base for THIS (kdi, khi, kwi) at this oc_blk:
                 *   step over (ic_blk, 4_cin, 4_cout) inner tiles. */
                const int w_kpos_base = w_oc_base
                                        + ((kdi * kh + khi) * kw + kwi) * in_cblock * 16;

                /* Loop over input-channel blocks. Each iteration:
                 *   load 4 input channels (vload4) + 16 weights (4 cin x 4 cout)
                 *   accumulate 4 MADs into the 4-channel accumulator. */
                for (int ic_blk = 0; ic_blk < in_cblock; ++ic_blk) {
                    /* Input layout: [cblock, batch=1, D, H, W, 4]
                     *   offset_voxel = (ic_blk * D + idz) * H * W + idy * W + idx
                     * (we treat batch=1 implicitly; the lower bits of
                     * the offset are inside the C/4 block). */
                    const int in_voxel = ((ic_blk * in_d + idz) * in_h + idy) * in_w + idx;
                    COMPUTE_FLOAT4 in_v = CONVERT_COMPUTE_FLOAT4(vload4(in_voxel, input));

                    /* Weights for this (oc_blk, kdi, khi, kwi, ic_blk).
                     * 4 float4s = 16 floats total. Each float4 is the
                     * 4 cout values for a single cin lane. */
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

    /* Output: same NC4DHW4 layout as input. */
    const int out_voxel = ((oc_blk * out_d + od) * out_h + oh) * out_w + ox;
    vstore4(CONVERT_FLOAT4(acc), out_voxel, output);
}

/*
 * conv_3d_buf_nchw_in
 *
 * Variant that reads its INPUT as plain unpacked NCHW (1, C, D*H*W)
 * instead of NC4HW4-packed. Used when the Conv3D input is a
 * MEMORY_VIRTUAL tensor with an identity-stride Region pointing at an
 * NCHW-formatted upstream tensor (MNN's BUFFER backend stores NCHW
 * tensors as unpacked [N*C*H*W] floats). Consuming the upstream
 * NCHW buffer directly lets us skip the Raster materialization step
 * MNN would otherwise schedule -- removing one huge (Cin*D*H*W*4)
 * intermediate buffer per Conv3D op.
 *
 * OUTPUT is still NC4DHW4 packed (matches the Conv3D output tensor's
 * declared format=NC4HW4).
 *
 * Cin is passed exactly (not rounded to multiple-of-4); the inner
 * loop reads `Cin` consecutive channels for each (kd,kh,kw) tile,
 * 4-per-iteration where possible, with a single-channel fallback for
 * the leftover at the end.
 *
 * Layout maps:
 *   input[c, d, h, w] = input[c * (in_d*in_h*in_w) + d*in_h*in_w + h*in_w + w]
 */
__kernel void conv_3d_buf_nchw_in(GLOBAL_SIZE_3_DIMS
                                   __global const FLOAT *input,
                                   __global const FLOAT *weight,
                                   __global const FLOAT *bias,
                                   __global FLOAT *output,
                                   __private const int in_c,
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

    const int in_d_base = od * sd - pd;
    const int in_h_base = oh * sh - ph;
    const int in_w_base = ox * sw - pw;

    /* in_cblock = number of 4-cin tiles, including a possibly-padded
     * tail block when in_c is not a multiple of 4. */
    const int in_cblock = (in_c + 3) >> 2;
    const int in_spatial = in_d * in_h * in_w;

    const int w_kvol   = kd * kh * kw;
    const int w_per_oc = w_kvol * in_cblock * 4 * 4;
    const int w_oc_base = oc_blk * w_per_oc;

    for (int kdi = 0; kdi < kd; ++kdi) {
        const int idz = in_d_base + kdi;
        if (idz < 0 || idz >= in_d) continue;
        for (int khi = 0; khi < kh; ++khi) {
            const int idy = in_h_base + khi;
            if (idy < 0 || idy >= in_h) continue;
            for (int kwi = 0; kwi < kw; ++kwi) {
                const int idx = in_w_base + kwi;
                if (idx < 0 || idx >= in_w) continue;

                const int w_kpos_base = w_oc_base
                                        + ((kdi * kh + khi) * kw + kwi) * in_cblock * 16;
                const int spat_off = (idz * in_h + idy) * in_w + idx;

                for (int ic_blk = 0; ic_blk < in_cblock; ++ic_blk) {
                    const int c_base = ic_blk * 4;

                    /* Read 4 channels (strided by in_spatial). The
                     * tail block reads zeros for c >= in_c. */
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

    const int out_voxel = ((oc_blk * out_d + od) * out_h + oh) * out_w + ox;
    vstore4(CONVERT_FLOAT4(acc), out_voxel, output);
}
