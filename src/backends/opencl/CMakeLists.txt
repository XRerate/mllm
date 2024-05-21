# CPU Backend
file(GLOB MLLM_OPENCL_SRC
        ${CMAKE_CURRENT_LIST_DIR}/*.cpp
        ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "found openmp")
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
else()
    message(FATAL_ERROR "openmp not found!")
endif()

find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "found opencl")
else()
    message(FATAL_ERROR "opencl not found!")
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86_64 detected")
add_compile_options(-mavx2)
add_compile_options(-march=native)
endif()

add_library(
        MLLM_OPENCL
        OBJECT
        ${MLLM_OPENCL_SRC}
)

target_include_directories(
        MLLM_OPENCL
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

if(OpenMP_FOUND)
    message(STATUS "found openmp")
    if(ARM AND NOT APK)
        message(STATUS "[ARM]  found openmp")
        target_compile_options(MLLM_OPENCL PRIVATE -fopenmp)
        target_link_libraries(MLLM_OPENCL PUBLIC -fopenmp -static-openmp)
        target_link_libraries(MLLM_OPENCL PUBLIC OpenCL::OpenCL)
    else()
    target_link_libraries(MLLM_OPENCL
            PUBLIC
            OpenMP::OpenMP_CXX 
            OpenCL::OpenCL)
    endif()
endif()