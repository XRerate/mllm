#ifndef MLLM_OPENCLGEMM_H
#define MLLM_OPENCLGEMM_H

#include "OpenCLOp.hpp"

namespace mllm {

class OpenCLGEMM final : public OpenCLOp {
public:
    OpenCLGEMM(Backend *bn, string opName) :
        OpenCLOp(bn, opName) {
    }
    virtual ~OpenCLGEMM() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    const char *getSrc() override {
        return
#include "backends/opencl/OpenCLGEMM.cl"
            ;
    }

private:
    int thread_count = 4;
};

class OpenCLGEMMCreator : public OpenCLBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        OpenCLOp *op = new OpenCLGEMM(bn, name);
        // TODO(seonjunn): this is lame
        op->createKernel();
        return op;
    }
};

} // namespace mllm

#endif // MLLM_OPENCLGEMM_H