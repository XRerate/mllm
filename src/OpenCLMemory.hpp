#pragma once

#include <cstddef>
#include <cstring>
#include <any>
#include "Memory.hpp"
#include "backends/opencl/OpenCLBackend.hpp"

namespace mllm {

class OpenCLMemory : public Memory {
public:
    OpenCLMemory(Backend *backend) : Memory(backend) {}
    ~OpenCLMemory() {
        backend_->free(device_ptr_);
    }

    void copyTo(Memory *dst, size_t size) override {
        backend_->copy(device_ptr_, dst->device_ptr(), size);
    }

    void allocDevice(size_t size) override {
        backend_->alloc(&device_ptr_, size, 8);
    }

    void deviceToHost(size_t offset, size_t size) override {
        backend_->deviceToHost(device_ptr_, host_ptr_, offset, size);
    }

    void hostToDevice(size_t offset, size_t size) override {
        backend_->hostToDevice(host_ptr_, device_ptr_, offset, size);
    }
};

} // namespace mllm

