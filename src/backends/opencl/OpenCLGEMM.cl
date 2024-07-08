#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
return;                                                                                   \
}

// From MNN/source/backend/opencl/core/runtime/OpenCLRuntime.cpp -> buildKernel function
#define FLOAT float
#define COMPUTE_FLOAT float
#define COMPUTE_FLOAT4 float4
#define CONVERT_FLOAT4 convert_float4
#define CONVERT_COMPUTE_FLOAT4 convert_float4

// input_a: M x K, input_b: K x N, output_c: M x N
// channels: K, channel_blocks: ceil(K/4)*4, width_blocks: ceil(N/4)*4, width: N
__kernel void matmul_buf(GLOBAL_SIZE_2_DIMS __global const FLOAT* input_a,
                     __global const FLOAT* input_b,
                     #ifdef BIAS
                     __global const FLOAT* input_c,
                     #endif
                     __global FLOAT* output_c, 
                     __private const int channels,
                     __private const int channel_blocks,
                     __private const int width_blocks,
                     __private const int width) {
    const int width_blocks_idx = get_global_id(0);// output W
    const int height_idx       = get_global_id(1);// output H

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    COMPUTE_FLOAT4 a;
    COMPUTE_FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    COMPUTE_FLOAT4 v_zero = (COMPUTE_FLOAT4)((COMPUTE_FLOAT)0.0);

    #ifdef BIAS
    COMPUTE_FLOAT4 temp = CONVERT_COMPUTE_FLOAT4(vload4(width_blocks_idx, input_c));

    COMPUTE_FLOAT result0 = temp.x;
    COMPUTE_FLOAT result1 = temp.y;
    COMPUTE_FLOAT result2 = temp.z;
    COMPUTE_FLOAT result3 = temp.w;
    #else
    COMPUTE_FLOAT result0 = 0;
    COMPUTE_FLOAT result1 = 0;
    COMPUTE_FLOAT result2 = 0;
    COMPUTE_FLOAT result3 = 0;
    #endif

    const int remain = channel_blocks*4 - channels;
    for (short pos = 0; pos < channel_blocks - 1; pos += 1) {
        const int inpa_offset = height_idx * channel_blocks + pos;
        a = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));

        const int inpb_offset = (pos*4) * width_blocks + width_blocks_idx;

        b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        b1 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks, input_b));
        b2 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*2, input_b));
        b3 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*3, input_b));

        COMPUTE_FLOAT4 btmp0 = (COMPUTE_FLOAT4)(b0.s0, b1.s0, b2.s0, b3.s0);
        COMPUTE_FLOAT4 btmp1 = (COMPUTE_FLOAT4)(b0.s1, b1.s1, b2.s1, b3.s1);
        COMPUTE_FLOAT4 btmp2 = (COMPUTE_FLOAT4)(b0.s2, b1.s2, b2.s2, b3.s2);
        COMPUTE_FLOAT4 btmp3 = (COMPUTE_FLOAT4)(b0.s3, b1.s3, b2.s3, b3.s3);

        result0 += dot(a, btmp0);
        result1 += dot(a, btmp1);
        result2 += dot(a, btmp2);
        result3 += dot(a, btmp3);
    }
    
    {
        const int inpa_offset = height_idx * channel_blocks + channel_blocks - 1;
        a = CONVERT_COMPUTE_FLOAT4(vload4(inpa_offset, input_a));

        const int inpb_offset = ((channel_blocks - 1)*4) * width_blocks + width_blocks_idx;

        b0 = CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset, input_b));
        b1 = (remain >= 3) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks, input_b));
        b2 = (remain >= 2) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*2, input_b));
        b3 = (remain >= 1) ? v_zero : CONVERT_COMPUTE_FLOAT4(vload4(inpb_offset + width_blocks*3, input_b));
        if (remain == 3) {
            a.y = 0;
            a.z = 0;
            a.w = 0;
        } else if (remain == 2) {
            a.z = 0;
            a.w = 0;
        } else if (remain == 1) {
            a.w = 0;;
        }

        COMPUTE_FLOAT4 btmp0 = (COMPUTE_FLOAT4)(b0.s0, b1.s0, b2.s0, b3.s0);
        COMPUTE_FLOAT4 btmp1 = (COMPUTE_FLOAT4)(b0.s1, b1.s1, b2.s1, b3.s1);
        COMPUTE_FLOAT4 btmp2 = (COMPUTE_FLOAT4)(b0.s2, b1.s2, b2.s2, b3.s2);
        COMPUTE_FLOAT4 btmp3 = (COMPUTE_FLOAT4)(b0.s3, b1.s3, b2.s3, b3.s3);

        result0 += dot(a, btmp0);
        result1 += dot(a, btmp1);
        result2 += dot(a, btmp2);
        result3 += dot(a, btmp3);
    }

    const int out_offset = height_idx * width_blocks + width_blocks_idx;
    vstore4(CONVERT_FLOAT4((COMPUTE_FLOAT4)(result0, result1, result2, result3)), out_offset, output_c);
}