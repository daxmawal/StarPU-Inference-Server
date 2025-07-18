set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(UTILS_DIR ${SRC_DIR}/utils)

add_executable(unit_tests test_datatype_utils.cpp test_time_utils.cpp
                          test_input_generator.cpp ${UTILS_DIR}/time_utils.cpp)

set_target_properties(unit_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                            ${CMAKE_BINARY_DIR})

target_compile_features(unit_tests PRIVATE cxx_std_23)
target_compile_definitions(starpu_server PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_include_directories(unit_tests PRIVATE ${SRC_DIR} ${UTILS_DIR})

target_link_libraries(unit_tests PRIVATE gtest gtest_main Torch::Torch
                                         StarPU::starpu)

add_test(NAME unit_tests COMMAND unit_tests)
