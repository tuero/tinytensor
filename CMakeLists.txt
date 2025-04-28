cmake_minimum_required(VERSION 3.25)

project(tinytensor CXX)

# Build options
option(TT_BUILD_CUDA "Build tinytensor with cuda backend support" OFF)
option(TT_BULD_TESTS "Build tinytensor tests" OFF)
option(TT_BULD_EXAMPLES "Build tinytensor examples" OFF)

# Find cuda and set flags to build cuda backend
set(TT_BUILD_CUDA_AND_FOUND OFF)
if(TT_BUILD_CUDA)
    find_package(CUDAToolkit)
    if(${CUDAToolkit_FOUND})
        
        enable_language(CUDA)
        set(TT_BUILD_CUDA_AND_FOUND ON)
        set(TINYTENSOR_CUDA_FLAGS "--expt-relaxed-constexpr")

        message("Cuda found, building with cuda backend support")
    else()
        message(WARNING "Couldn't find cuda, skipping cuda backend support")
    endif()
endif()

# Enable SSE
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Library source
add_subdirectory(tinytensor)

# Build tests
if (${TT_BUILD_TESTS})
    enable_testing()
    add_subdirectory(tests)
endif()

# Build examples
if (${TT_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()
