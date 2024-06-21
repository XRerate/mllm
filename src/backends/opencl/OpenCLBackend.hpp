#ifndef MLLM_OPENCLBACKEND_H
#define MLLM_OPENCLBACKEND_H

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include "Backend.hpp"
#include "../../OpenCLMemory.hpp"
#include "Op.hpp"
#include "Types.hpp"
//#include "CL/cl.h"
#include "OpenCLRuntime.hpp"
#include <filesystem>

namespace mllm {

class OpenCLBackend final : public Backend {
public:
    explicit OpenCLBackend(shared_ptr<MemoryManager> &mm);
    ~OpenCLBackend() override;

    class Creator {
    public:
        virtual Op *create(OpParam op_param, Backend *bn, string, int threadCount) const = 0;
    };
    bool addCreator(OpType t, Creator *c) {
        if (map_creator_.find(t) != map_creator_.end()) {
            printf("Error: %d type has be added\n", t);
            return false;
        }
        map_creator_.insert(std::make_pair(t, c));
        return true;
    }

    void initOpenCL();

    Memory *createMemory() override {
        return new OpenCLMemory(this);
    }

    void alloc(void **ptr, size_t size, size_t alignment) override {
        cl_int ret;
        *ptr = opencl::clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
        assert(ret == CL_SUCCESS);
    }
    void free(void *ptr) override {
        cl_int ret = opencl::clReleaseMemObject(static_cast<cl_mem>(ptr));
        assert(ret == CL_SUCCESS);
    }

    void copy(void *src, void *dst, size_t size) override {
        cl_int ret = opencl::clEnqueueCopyBuffer(command_queue, static_cast<cl_mem>(src), static_cast<cl_mem>(dst),
                                                 0, 0, size, 0, NULL, NULL);
        assert(ret == CL_SUCCESS);
    }
    void deviceToHost(void *device_ptr, void *host_ptr, size_t offset, size_t size) override {
        char *host_ptr_offset = static_cast<char *>(host_ptr) + offset;
        cl_int ret = opencl::clEnqueueReadBuffer(command_queue, static_cast<cl_mem>(device_ptr),
                                                 CL_TRUE, offset, size, host_ptr_offset, 0, NULL, NULL);
        assert(ret == CL_SUCCESS);
    }
    void hostToDevice(void *host_ptr, void *device_ptr, size_t offset, size_t size) override {
        char *host_ptr_offset = static_cast<char *>(host_ptr) + offset;
        cl_int ret = opencl::clEnqueueWriteBuffer(command_queue, static_cast<cl_mem>(device_ptr),
                                                  CL_TRUE, offset, size, host_ptr_offset, 0, NULL, NULL);
        assert(ret == CL_SUCCESS);
    }

    Op *opCreate(const OpParam &op_param, string name, int threadCount) override;
    TensorFunction *funcCreate(const TensorFuncType type) override;

    void registerOps() override;
    void registerFuncs() override;

    static int cpu_threads;

    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device_id;

private:
    std::map<OpType, OpenCLBackend::Creator *> map_creator_;
    std::map<TensorFuncType, TensorFunction *> map_function_;
};

} // namespace mllm

#endif // MLLM_OPENCLBACKEND_H