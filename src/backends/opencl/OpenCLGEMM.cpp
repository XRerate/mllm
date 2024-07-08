
#include "OpenCLGEMM.hpp"

namespace mllm {

ErrorCode OpenCLGEMM::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    if (inputs[0]->batch() == 1 || inputs[1]->batch() == 1) {
    } else {
        assert(inputs[0]->batch() == inputs[1]->batch());
    }
    
    // TODO: Do similar thing with CPUMatmul

    return Op::reshape(inputs, outputs);
}

ErrorCode OpenCLGEMM::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    OpenCLBackend *opencl_bn = dynamic_cast<OpenCLBackend *>(backend());
    cl_mem input0 = static_cast<cl_mem>(inputs[0]->devicePtr());
    cl_mem input1 = static_cast<cl_mem>(inputs[1]->devicePtr());
    cl_mem output = static_cast<cl_mem>(outputs[0]->devicePtr());

    // TODO: Fix this

    cl_int ret;
    ret = opencl::clSetKernelArg(kernel_, 0, sizeof(cl_mem), &input0);
    ret = opencl::clSetKernelArg(kernel_, 1, sizeof(cl_mem), &input1);
    ret = opencl::clSetKernelArg(kernel_, 2, sizeof(cl_mem), &output);

    inputs[0]->dataPtr()->deviceToHost();

    size_t global_work_size = inputs[0]->count();
    ret = opencl::clEnqueueNDRangeKernel(opencl_bn->command_queue, kernel_, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

    outputs[0]->dataPtr()->deviceToHost();

    return Op::execute(inputs, outputs);
}
} // namespace mllm
