#ifndef MLLM_OPENCLBACKEND_H
#define MLLM_OPENCLBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "CL/cl.h"

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
    Op *opCreate(const OpParam &op_param, string name, int threadCount) override;
    TensorFunction *funcCreate(const TensorFuncType type) override;

    void registerOps() override;
    void registerFuncs() override;

    static int cpu_threads;

    cl_context context;
    cl_command_queue command_queue;
private:
    std::map<OpType, OpenCLBackend::Creator *> map_creator_;
    std::map<TensorFuncType, TensorFunction *> map_function_;
};

} // namespace mllm

#endif // MLLM_OPENCLBACKEND_H