#include "OpenCLBackend.hpp"
#include "OpenCLAdd.hpp"
#include <iostream>
#include <math.h>

namespace mllm {
OpenCLBackend::OpenCLBackend(shared_ptr<MemoryManager> &mm) :
    Backend(mm) {
    initOpenCL();
    registerOps();
    registerFuncs();
}

OpenCLBackend::~OpenCLBackend() {
    cl_int ret;
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
}

void OpenCLBackend::initOpenCL() {
    cl_int ret;
    cl_uint num_platforms;
    cl_platform_id platform_id;
    ret = clGetPlatformIDs(1, &platform_id, &num_platforms);
    assert(ret == CL_SUCCESS);
    cl_uint num_devices;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    assert(ret == CL_SUCCESS);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    assert(ret == CL_SUCCESS);
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    assert(ret == CL_SUCCESS);
}

Op *OpenCLBackend::opCreate(const OpParam &op_param, string name, int threadCount) {
    OpType optype = OpType(op_param.find("type")->second);
    auto iter = map_creator_.find(optype);
    if (iter == map_creator_.end()) {
        printf("Don't support type \n");
        return nullptr;
    }
    Op *exe = iter->second->create(op_param, this, name, 0); // TODO: Remove threadCount parameter for opencl
    return exe;
}
void OpenCLBackend::registerOps() {
    addCreator(ADD, (OpenCLBackend::Creator *)(new OpenCLAddCreator()));
}

TensorFunction *OpenCLBackend::funcCreate(const TensorFuncType type) {
    auto iter = map_function_.find(type);
    if (iter == map_function_.end()) {
        printf("Don't support type \n");
        return nullptr;
    }
    return iter->second;
}

void OpenCLBackend::registerFuncs() {
    // map_function_[TensorFuncType::FUNC_ADD] = new OpenCLaddFunction();
};

} // namespace mllm
