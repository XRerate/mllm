#ifndef MLLM_BACKEND_H
#define MLLM_BACKEND_H

#include "Memory.hpp"
#include "MemoryManager.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
#include <memory>
using std::shared_ptr;

namespace mllm {
class Op;

class Tensor;
class Memory;



class TensorFunction{
public:
    virtual void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) = 0;
    virtual void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) = 0;
};
class Backend {
public:
    Backend(shared_ptr<MemoryManager> &mm) :
        mem_manager_(mm) {
    }
    virtual ~Backend() = default;

    /**
     * \brief Allocates memory of the given size and alignment.
     * \param ptr A pointer to the pointer where the start address of the allocated memory will be stored.
     * \param size The size of the memory to be allocated.
     * \param alignment The alignment of the memory to be allocated.
     */
    virtual void alloc(void **ptr, size_t size, size_t alignment) {
        mem_manager_->alloc(ptr, size, alignment);
    }

    /**
     * \brief Frees the memory pointed to by ptr.
     * \param ptr A pointer to the memory to be freed.
     */
    virtual void free(void *ptr) {
        mem_manager_->free(ptr);
    }

    virtual Memory *createMemory() = 0;
    virtual void copy(void *src, void *dst, size_t size) = 0;
    virtual void deviceToHost(void *device_ptr, void *host_ptr, size_t offset, size_t size) {};
    virtual void hostToDevice(void *host_ptr, void *device_ptr, size_t offset, size_t size) {};

    /**
     * \brief Creates an operation(Op) with the given parameters.
     * \param op_param The parameters for the operation to be created.
     * \param name The name of the operation. Default is an empty string.
     * \param threadCount The number of threads to be used for the operation. Default is 4.
     * \return A pointer to the created operation.
     */
    virtual Op *opCreate(const OpParam &op_param, string name = "", int threadCount = 4) = 0;
    virtual TensorFunction *funcCreate(const TensorFuncType type) = 0;

    /**
     * \brief Registers all the operations supported by the backend.
     * This function is expected to be overridden by each specific backend implementation.
     */
    virtual void registerOps() = 0;
    virtual void registerFuncs() = 0;

private:
    shared_ptr<MemoryManager> mem_manager_;
};

} // namespace mllm

#endif // MLLM_BACKEND_H