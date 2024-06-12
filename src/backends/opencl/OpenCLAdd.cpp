
#include "OpenCLAdd.hpp"

namespace mllm {

ErrorCode OpenCLAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    if (inputs[0]->batch() == 1 || inputs[1]->batch() == 1) {
    } else {
        assert(inputs[0]->batch() == inputs[1]->batch());
    }
    assert(inputs[0]->head() == inputs[1]->head());
    assert(inputs[0]->sequence() == inputs[1]->sequence());
    assert(inputs[0]->dimension() == inputs[1]->dimension());
    outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode OpenCLAdd::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    OpenCLBackend *opencl_bn = dynamic_cast<OpenCLBackend *>(backend());
    cl_mem input0 = static_cast<cl_mem>(inputs[0]->devicePtr());
    cl_mem input1 = static_cast<cl_mem>(inputs[1]->devicePtr());
    cl_mem output = static_cast<cl_mem>(outputs[0]->devicePtr());

    cl_int ret;
    ret = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &input0);
    ret = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &input1);
    ret = clSetKernelArg(kernel_, 2, sizeof(cl_mem), &output);

    inputs[0]->dataPtr()->deviceToHost();

    size_t global_work_size = inputs[0]->count();
    ret = clEnqueueNDRangeKernel(opencl_bn->command_queue, kernel_, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

    outputs[0]->dataPtr()->deviceToHost();

    return Op::execute(inputs, outputs);
}
} // namespace mllm
