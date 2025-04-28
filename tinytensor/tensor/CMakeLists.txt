target_sources(tinytensor PRIVATE
    tensor.cpp
    backend_base.h
    backend_register.cpp
    backend_register.h
    random.cpp
    index.cpp
    print.h
    shape.cpp
    storage_base.h
)
# If building with CUDA support, some common files need to be built in CUDA 
#  mode instead to ensure linker can find the host/device methods
if(${TT_BUILD_CUDA_AND_FOUND})
    set_source_files_properties(random.cpp TARGET_DIRECTORY tinytensor PROPERTIES LANGUAGE CUDA) 
endif()


# Add backends
add_subdirectory(backend/common)
add_subdirectory(backend/cpu)
if(${TT_BUILD_CUDA_AND_FOUND})
    target_compile_features(tinytensor PUBLIC cuda_std_20)
    target_compile_options(tinytensor PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${TINYTENSOR_CUDA_FLAGS}>")
    target_link_libraries(tinytensor CUDA::cudart)
    set_target_properties(tinytensor PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
    )
    add_subdirectory(backend/cuda) 
endif()

