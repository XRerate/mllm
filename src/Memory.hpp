#pragma once

#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <any>

namespace mllm {

class Backend;

class Memory {
public:
    Memory(Backend *backend) : backend_(backend) {}
    virtual ~Memory() {
        free(host_ptr_);
    }

    void alloc(size_t size) {
        size_ = size;
        host_ptr_ = aligned_alloc(8, size);
        assert(host_ptr_ != nullptr);
        allocDevice(size);
    }

    template <typename Dtype>
    Dtype *ptrAt(size_t offset) {
        deviceToHost(); // copy all
        return static_cast<Dtype*>(host_ptr_) + offset*sizeof(Dtype);
    }
    
    template <typename Dtype>
    Dtype dataAt(size_t offset) {
        deviceToHost(offset*sizeof(Dtype), sizeof(Dtype));
        return static_cast<Dtype*>(host_ptr_)[offset];
    }

    template <typename Dtype>
    void setDataAt(size_t offset, Dtype value) {
        static_cast<Dtype*>(host_ptr_)[offset] = value;
        hostToDevice(offset*sizeof(Dtype), sizeof(Dtype));
    }

    virtual void copyTo(Memory *dst, size_t size) = 0;

    virtual void deviceToHost(size_t offset, size_t size) {};
    virtual void deviceToHost() { deviceToHost(0, size_); }
    virtual void hostToDevice(size_t offset, size_t size) {};
    virtual void hostToDevice() { hostToDevice(0, size_); }

    void *host_ptr() { return host_ptr_; }
    void *device_ptr() { return device_ptr_; }

protected:
    void *host_ptr_;
    void *device_ptr_;
    Backend *backend_;
    size_t size_;

private:
    virtual void allocDevice(size_t size) {};
};
} // namespace mllm

