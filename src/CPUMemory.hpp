#pragma once

#include <cstddef>
#include <cstring>
#include <any>
#include "Backend.hpp"
#include "Memory.hpp"
#include "backends/opencl/OpenCLBackend.hpp"

namespace mllm {

class Backend;

class CPUMemory : public Memory {
public:
    CPUMemory(Backend *backend) : Memory(backend) {}
    ~CPUMemory() {}

    void copyTo(Memory *dst, size_t size) override {
        backend_->copy(host_ptr_, dst->host_ptr(), size);
    }
};

} // namespace mllm

