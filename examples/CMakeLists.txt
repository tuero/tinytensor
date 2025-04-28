function(CREATE_EXAMPLE EXAMPLE_NAME)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}.cpp)
    target_link_libraries(${EXAMPLE_NAME} tinytensor::tinytensor)
endfunction()

CREATE_EXAMPLE(mlp_example)
CREATE_EXAMPLE(mnist_cnn_example)
CREATE_EXAMPLE(translation_example)
