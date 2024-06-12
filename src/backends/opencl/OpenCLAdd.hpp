#ifndef MLLM_OPENCLADD_H
#define MLLM_OPENCLADD_H

#include "OpenCLOp.hpp"

namespace mllm {

class OpenCLAdd final : public OpenCLOp {
public:
    OpenCLAdd(Backend *bn, string opName): OpenCLOp(bn, opName, "backends/opencl/OpenCLAdd.cl") {}
    virtual ~OpenCLAdd() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;


private:
    int thread_count = 4;
};

class OpenCLAddCreator : public OpenCLBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new OpenCLAdd(bn, name);
    }
};

} // namespace mllm

#endif // MLLM_OPENCLADD_H