add_executable(unit_tests
    test_datatype_utils.cpp
    test_time_utils.cpp
    test_input_generator.cpp
    ${UTILS_DIR}/time_utils.cpp
)

set_target_properties(unit_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

target_compile_features(unit_tests PRIVATE cxx_std_23)

target_include_directories(unit_tests PRIVATE
    ${SRC_DIR}
    ${UTILS_DIR}
    ${STARPU_DIR}/include/starpu/1.4
)

target_link_libraries(unit_tests PRIVATE
    gtest
    gtest_main
    ${TORCH_LIBRARIES}
)

add_test(NAME unit_tests COMMAND unit_tests)