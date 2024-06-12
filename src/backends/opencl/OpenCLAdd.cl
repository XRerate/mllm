__kernel void add(
    __global const float *in1, 
	__global const float *in2,
	__global float *out) {
		int gid = get_global_id(0);
		out[gid] = in1[gid] + in2[gid];
}