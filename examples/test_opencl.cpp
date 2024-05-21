#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);
    int thread_num = cmdParser.get<int>("thread");

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();
    auto *i1 = _Input(c);
    auto *i2 = _Input(c);
    auto *o1 = _Add({i1, i2});

    BackendConfig bn;
    Net net(bn);
    net.convert(c->sub_param_, BackendType::MLLM_OPENCL, thread_num);

    ParamLoader param_loader("../models/llama-2-7b-chat-q4_k.mllm");
    Executor ex(&param_loader);
    // ex.setup(&net);

    shared_ptr<Tensor> input1 = std::make_shared<Tensor>(std::vector<int>{2,2,1,1}); // BSHD. Always need 4 dimensions
    shared_ptr<Tensor> input2 = std::make_shared<Tensor>(std::vector<int>{2,2,1,1});
    input1->setDtype(MLLM_TYPE_F32);
    input2->setDtype(MLLM_TYPE_F32);
    input1->setBackend(net.backends()[BackendType::MLLM_OPENCL].get());
    input2->setBackend(net.backends()[BackendType::MLLM_OPENCL].get());
    input1->alloc();
    input2->alloc();
    input1->setDataAt<float>(0, 0, 0, 0, 1.);
    input1->setDataAt<float>(0, 0, 1, 0, 2.);
    input1->setDataAt<float>(1, 0, 0, 0, 3.);
    input1->setDataAt<float>(1, 0, 1, 0, 4.);
    
    input1->printData<float>();

    ex.run(&net, {input1, input2});
    auto result = ex.result()[0];
    result->printData<float>();

    ex.perf();

    // free memory
    for (auto *op : c->net_ops) {
        delete op;
    }
    for (auto *tensor : c->net_tensors) {
        delete tensor;
    }
    return 0;
}
