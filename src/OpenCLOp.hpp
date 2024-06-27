#pragma once

#include "Op.hpp"
#include "OpenCLBackend.hpp"
#include <CL/cl.h>

namespace mllm {

class OpenCLOp : public Op {
public:
    OpenCLOp(Backend *bn, std::string opName) :
        Op(bn, opName) {
    }

    ~OpenCLOp() {
        opencl::clReleaseKernel(kernel_);
        opencl::clReleaseProgram(program_);
    }

    virtual const char *getSrc() = 0;

    void createKernel() {
        // TODO: Remove dynamic_cast
        OpenCLBackend *opencl_bn = dynamic_cast<OpenCLBackend *>(backend());

        const char *kernelSource = getSrc();
        std::cout << kernelSource << std::endl;

        size_t kernelSize = strlen(kernelSource);
        std::cout << kernelSize << std::endl;
        cl_int ret;
        program_ = opencl::clCreateProgramWithSource(opencl_bn->context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &ret);
        ret = opencl::clBuildProgram(program_, 1, &opencl_bn->device_id, NULL, NULL, NULL);
        assert(ret == CL_SUCCESS);
        kernel_ = opencl::clCreateKernel(program_, "add", &ret);
        assert(ret == CL_SUCCESS);
    }

protected:
    cl_program program_;
    cl_kernel kernel_;
};

} // namespace mllm