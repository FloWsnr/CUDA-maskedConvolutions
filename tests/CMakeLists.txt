include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#######################################################
# Add the tests #######################################
#######################################################

add_executable(tests test_mConv1D.cu test_mConv1D_cpu.cpp)

target_link_libraries(
  tests PRIVATE
  GTest::gtest_main
  ${LIB_NAME}
)
# target_link_libraries(test_masked_conv PRIVATE masked_conv)

set_target_properties(tests PROPERTIES
CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(tests PROPERTIES CUDA_ARCHITECTURES 75)


include(GoogleTest)
gtest_discover_tests(tests)