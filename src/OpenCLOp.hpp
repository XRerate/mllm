#pragma once

#include "Op.hpp"
#include "OpenCLBackend.hpp"
#include <CL/cl.h>

namespace mllm {

class OpenCLOp : public Op {
public:
    OpenCLOp(Backend *bn, std::string opName, std::string kernelSourceFilePath) : 
        Op(bn, opName), kernelSourceFilePath(kernelSourceFilePath) {
        createKernel();
    }

    ~OpenCLOp() {
        clReleaseKernel(kernel_);
        clReleaseProgram(program_);
    }

protected:
    cl_program program_;
    cl_kernel kernel_;

    std::string kernelSourceFilePath;

    void createKernel() {
        // TODO: Remove dynamic_cast
        OpenCLBackend *opencl_bn = dynamic_cast<OpenCLBackend *>(backend());
        std::filesystem::path currentDir = std::filesystem::path(__FILE__).parent_path();

        std::cout << (currentDir / kernelSourceFilePath).string() << std::endl;
        std::ifstream kernelFile((currentDir / kernelSourceFilePath).string());
        assert (kernelFile.is_open());
        std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
        const char *kernelSourceChar = kernelSource.c_str();
        size_t kernelSize = kernelSource.size();
        cl_int ret;
        program_ = clCreateProgramWithSource(opencl_bn->context, 1, (const char **)&kernelSourceChar, (const size_t *)&kernelSize, &ret);
        ret = clBuildProgram(program_, 1, &opencl_bn->device_id, NULL, NULL, NULL);
        assert(ret == CL_SUCCESS);
        kernel_ = clCreateKernel(program_, "add", &ret);
        assert(ret == CL_SUCCESS);
    }

};

} // namespace mllm