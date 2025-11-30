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

# Add backends
add_subdirectory(backend/common)
add_subdirectory(backend/cpu)
if(${TT_BUILD_CUDA_AND_FOUND})
    add_subdirectory(backend/cuda) 
    # If building with CUDA support, some common files need to be built in CUDA 
    #  mode instead to ensure linker can find the host/device methods
    set_source_files_properties(random.cpp TARGET_DIRECTORY tinytensor PROPERTIES LANGUAGE CUDA) 
endif()

