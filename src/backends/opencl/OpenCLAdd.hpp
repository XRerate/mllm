#ifndef MLLM_OPENCLADD_H
#define MLLM_OPENCLADD_H

#include "OpenCLOp.hpp"

namespace mllm {

class OpenCLAdd final : public OpenCLOp {
public:
    OpenCLAdd(Backend *bn, string opName) :
        OpenCLOp(bn, opName) {
    }
    virtual ~OpenCLAdd() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    const char *getSrc() override {
        return
#include "backends/opencl/OpenCLAdd.cl"
            ;
    }

private:
    int thread_count = 4;
};

class OpenCLAddCreator : public OpenCLBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        OpenCLOp *op = new OpenCLAdd(bn, name);
        // TODO(seonjunn): this is lame
        op->createKernel();
        return op;
    }
};

} // namespace mllm

#endif // MLLM_OPENCLADD_H