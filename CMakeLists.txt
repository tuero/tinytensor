cmake_minimum_required(VERSION 3.25)

project(tinytensor VERSION 1.0.0 LANGUAGES CXX)

# Build options
option(TT_BUILD_CUDA "Build tinytensor with cuda backend support" OFF)
option(TT_BULD_TESTS "Build tinytensor tests" OFF)
option(TT_BULD_EXAMPLES "Build tinytensor examples" OFF)

# Hide symbols by default
if (NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET AND
    NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
endif ()

# Let tinytensor_SHARED_LIBS override BUILD_SHARED_LIBS
if (DEFINED tinytensor_SHARED_LIBS)
    set(BUILD_SHARED_LIBS "${tinytensor_SHARED_LIBS}")
endif ()

# Library and properties
add_library(tinytensor)
add_library(tinytensor::tinytensor ALIAS tinytensor)
set_target_properties(tinytensor PROPERTIES
                      VERSION ${tinytensor_VERSION}
                      SOVERSION ${tinytensor_VERSION_MAJOR}
                      POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    tinytensor PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_compile_features(tinytensor PUBLIC cxx_std_20)

# Generate export headers
include(GenerateExportHeader)
generate_export_header(tinytensor 
    EXPORT_FILE_NAME include/tt/export.h
)
target_compile_definitions(
    tinytensor PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:TINYTENSOR_STATIC_DEFINE>")
target_include_directories(
    tinytensor PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

# Find cuda and set flags to build cuda backend
set(TT_BUILD_CUDA_AND_FOUND OFF)
if(TT_BUILD_CUDA)
    find_package(CUDAToolkit)
    if(${CUDAToolkit_FOUND})
        message("Cuda found, building with cuda backend support")
        set(TT_BUILD_CUDA_AND_FOUND ON)
        enable_language(CUDA)
        set(TINYTENSOR_CUDA_FLAGS "--expt-relaxed-constexpr")
        target_compile_features(tinytensor PUBLIC cuda_std_20)
        target_compile_options(tinytensor PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${TINYTENSOR_CUDA_FLAGS}>")
        target_link_libraries(tinytensor PRIVATE CUDA::cudart)
        set_target_properties(tinytensor PROPERTIES 
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        # Macro definition so source files can conditionally set CUDA device/backend
        target_compile_definitions(tinytensor PUBLIC TT_CUDA)

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

# Include the install rules if the user wanted them (included by default when top-level)
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)
option(tinytensor_INCLUDE_PACKAGING "Include packaging rules for tinytensor" "${is_top_level}")
if (tinytensor_INCLUDE_PACKAGING)
    add_subdirectory(packaging)
endif ()

# Build tests
if (${TT_BUILD_TESTS})
    enable_testing()
    add_subdirectory(tests)
endif()

# Build examples
if (${TT_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()
