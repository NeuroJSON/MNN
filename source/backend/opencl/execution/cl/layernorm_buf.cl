#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void layernorm_buf(__private int global_dim0, __private int global_dim1,
                        __global const FLOAT * input,
                        __global FLOAT * output,
                        __private const int inside,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
#if LOCAL_SIZE > 1
    float local sum_mnn[LOCAL_SIZE];
    #ifndef RMSNORM
    float local sum_mean_mnn[LOCAL_SIZE];
    #endif
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int lid = get_local_id(0);
        const int offset = pos.y * inside;
        const int inside_v4 = (inside + 3) >> 2;
        #ifdef PACK_LEAVE
        const int loop = inside_v4 - 1;
        const int inside_remain = inside - ((inside_v4-1) << 2);
        #else
        const int loop = inside_v4;
        #endif
        
        float4 in_sum = 0;
        int index = lid;
        #ifdef RMSNORM
        float4 mean = (float4)0;
        #else
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            in_sum += in;
        }
        sum_mean_mnn[lid] = in_sum.x + in_sum.y + in_sum.z+ in_sum.w;
        
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i){
                float in = input[offset + index * 4 + i];
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + in;
            }
        }
        #endif
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + sum_mean_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum_mean_mnn[0] / (float4)inside;
        #endif

        in_sum = 0;
        index = lid;
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            in_sum += (in - mean) * (in - mean);
        }
        sum_mnn[lid] = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i) {
                float in = input[offset + index * 4 + i];
                in = (in - mean.x) * (in - mean.x);
                sum_mnn[lid] = sum_mnn[lid] + in;
            }
        }
        #endif
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum_mnn[0] / (float4)inside;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        index = lid;
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            #ifdef GAMMA_BETA
            float4 out = (in - mean) * value * convert_float4(vload4(index, gamma)) + convert_float4(vload4(index, beta));
            #else
            float4 out = (in - mean) * value;
            #endif
            vstore4(CONVERT_FLOAT4(out), index, output + offset);
        }
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i){
                float in = input[offset + index * 4 + i];
                #ifdef GAMMA_BETA
                float out = (in - mean.x) * value.x * (float)gamma[index * 4 + i] + (float)beta[index * 4 + i];
                #else
                float out = (in - mean.x) * value.x;
                #endif
                output[offset + index * 4 + i] = out;
            }
        }
        #endif
    }
#else
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int offset = pos.y * inside;
        float in_sum = 0;
        #ifdef RMSNORM
        float mean = 0;
        #else
        for(int index = 0; index < inside; index++){
            in_sum += (float)input[offset + index];
        }
        float mean = in_sum / inside;
        #endif

        in_sum = 0;
        for(int index = 0; index < inside; index++){
            float in = (float)input[offset + index];
            in_sum += (in - mean) * (in - mean);
        }
        float square_sum = in_sum / inside;
        float value = 1.0f / sqrt(square_sum + epsilon);
        for(int i = 0; i < inside; ++i){
            float in = input[offset + i];
            #ifdef GAMMA_BETA
            float out = (in - mean) * value * (float)gamma[i] + (float)beta[i];
            #else
            float out = (in - mean) * value;
            #endif
            output[offset + i] = out;
        }
    }

#endif
}

/*
 * Split-row LayerNorm for few, very long rows (e.g. InstanceNorm on a 3D
 * volume reshaped to (1, C, D*H*W): 32 rows of 4.7M values). layernorm_buf runs
 * one work-group per row, which leaves most of a large GPU idle. Here:
 *   layernorm_split_stats    one work-group per (row, chunk): chunk count, mean
 *                            and M2 (sum of squared deviations from the chunk mean)
 *   layernorm_split_finalize one work-item per row: merge the chunks (Chan et al.)
 *                            into mean and 1/sqrt(var + eps)
 *   layernorm_split_norm     one work-item per float4 of the output
 * Requires inside % 4 == 0 (rows start on a float4 boundary).
 */
#ifndef SPLIT_LOCAL
#define SPLIT_LOCAL 256
#endif

__kernel void layernorm_split_stats(__global const FLOAT4* restrict input,
                                    __global float* restrict stats,
                                    __private const int inside4,
                                    __private const int chunk4,
                                    __private const int chunks) {
    __local float red[SPLIT_LOCAL];
    const int lid   = get_local_id(0);
    const int chunk = get_group_id(0);
    const int row   = get_global_id(1);
    const int begin = chunk * chunk4;
    const int end   = min(begin + chunk4, inside4);
    __global const FLOAT4* p = input + (size_t)row * inside4;
    const float n = (float)((end - begin) * 4);
#ifdef RMSNORM
    const float mean = 0.0f;
#else
    float4 s = (float4)0;
    for (int i = begin + lid; i < end; i += SPLIT_LOCAL) {
        s += convert_float4(p[i]);
    }
    red[lid] = s.x + s.y + s.z + s.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = SPLIT_LOCAL / 2; k > 0; k >>= 1) {
        if (lid < k) red[lid] += red[lid + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = red[0] / n;
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    float4 q = (float4)0;
    for (int i = begin + lid; i < end; i += SPLIT_LOCAL) {
        const float4 d = convert_float4(p[i]) - (float4)mean;
        q += d * d;
    }
    red[lid] = q.x + q.y + q.z + q.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = SPLIT_LOCAL / 2; k > 0; k >>= 1) {
        if (lid < k) red[lid] += red[lid + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        __global float* o = stats + ((size_t)row * chunks + chunk) * 3;
        o[0] = n;
        o[1] = mean;
        o[2] = red[0];
    }
}

__kernel void layernorm_split_finalize(__global const float* restrict stats,
                                       __global float* restrict rowstat,
                                       __private const int chunks,
                                       __private const int rows,
                                       __private const float epsilon) {
    const int row = get_global_id(0);
    if (row >= rows) return;
    __global const float* s = stats + (size_t)row * chunks * 3;
    float n = 0.0f, mean = 0.0f, m2 = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float nb = s[c * 3], mb = s[c * 3 + 1], m2b = s[c * 3 + 2];
        const float tot = n + nb;
        const float delta = mb - mean;
        mean += delta * (nb / tot);
        m2 += m2b + delta * delta * (n * nb / tot);
        n = tot;
    }
    rowstat[row * 2]     = mean;
    rowstat[row * 2 + 1] = 1.0f / sqrt(m2 / n + epsilon);
}

__kernel void layernorm_split_norm(__private const int global_dim0,
                                   __private const int global_dim1,
                                   __global const FLOAT4* restrict input,
                                   __global FLOAT4* restrict output,
                                   __global const float* restrict rowstat,
#ifdef GAMMA_BETA
                                   __global const FLOAT4* restrict gamma,
                                   __global const FLOAT4* restrict beta,
#endif
                                   __private const int inside4) {
    const int x = get_global_id(0);
    const int row = get_global_id(1);
    if (x >= global_dim0 || row >= global_dim1) return;
    const float mean = rowstat[row * 2];
    const float r = rowstat[row * 2 + 1];
    const size_t i = (size_t)row * inside4 + x;
    float4 v = (convert_float4(input[i]) - (float4)mean) * (float4)r;
#ifdef GAMMA_BETA
    v = v * convert_float4(gamma[x]) + convert_float4(beta[x]);
#endif
    output[i] = CONVERT_FLOAT4(v);
}
