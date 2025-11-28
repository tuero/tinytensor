file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${tinytensor_SOURCE_DIR}/include/tt/*.h")

add_library(tinytensor STATIC ${HEADER_LIST}
        ../include/tt/jit/Ops.h
        ../include/tt/jit/Graph.h
        ../include/tt/jit/OpNode.h
        tensor/backend/jit/storage_jit.h
        tensor/backend/jit/backend_jit.h
        tensor/backend/jit/backend_jit.cpp)
add_library(tinytensor::tinytensor ALIAS tinytensor)

set_target_properties(tinytensor PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(tinytensor PUBLIC cxx_std_20)
target_include_directories(tinytensor PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(tinytensor PRIVATE ${PROJECT_SOURCE_DIR}/tinytensor)
target_include_directories(tinytensor SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/libnop/include)

# Cuda macro defn on target so it gets inherited
if(${TT_BUILD_CUDA_AND_FOUND}) 
    target_compile_definitions(tinytensor PUBLIC TT_CUDA)
endif()

add_subdirectory(tensor)
add_subdirectory(common)
add_subdirectory(autograd)
add_subdirectory(nn)
add_subdirectory(optim)
add_subdirectory(data)

