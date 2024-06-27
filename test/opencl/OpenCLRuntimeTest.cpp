#include "gtest/gtest.h"
#include "backends/opencl/OpenCLRuntime.hpp"

namespace mllm {

TEST(OpenCLRuntimeTest, OpenCLRuntimeTest1) {
    EXPECT_EQ(opencl::LoadOpenCL(), true);
    // TODO(seonjunn): Add more tests (e.g., clGetPlatformInfo...)
}
} // namespace mllm