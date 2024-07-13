add_executable(tests tests.cu)
target_link_libraries(tests PRIVATE masked_conv)

set_target_properties(tests PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES 75)