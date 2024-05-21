
#include "memory/OpenCLSystemMemoryManager.hpp"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <malloc.h>

namespace mllm {

void OpenCLSystemMemoryManager::alloc(void **ptr, size_t size,size_t alignment){
    assert(size > 0);
    // TODO: Fix this to allocate opencl memory. Will need context variable in OpenCLBackend
    void **origin = (void **)malloc(size + sizeof(void *) + alignment-1);
    assert(origin != nullptr);
    if (origin == nullptr) {
        *ptr = nullptr;
    }
    void **aligned = (void**)(((size_t)(origin) + sizeof(void*) +  alignment - 1) & (~(alignment - 1)));
    aligned[-1]    = origin;
    *ptr = aligned;
}

void OpenCLSystemMemoryManager::free(void *ptr){
    // TODO: Fix this to free opencl memory.
    if (ptr != nullptr && malloc_usable_size(((void**)ptr)[-1]) > 0) {
        ::free(((void**)ptr)[-1]);
    }
}

} // namespace mllm