#ifndef MLLM_OPENCL_MEMORY_SYSTEM_H
#define MLLM_OPENCL_MEMORY_SYSTEM_H

#include "MemoryManager.hpp"
namespace mllm {
    class OpenCLSystemMemoryManager : public MemoryManager {
    public:
        OpenCLSystemMemoryManager(){}
        ~OpenCLSystemMemoryManager(){}

        void alloc(void **ptr, size_t size,size_t alignment) override ;

        void free(void *ptr) override;
    };
}
#endif