add_library(e2e_tests INTERFACE)
target_sources(e2e_tests INTERFACE smoke/test_full_inference.cpp
                                   regression/test_race_shutdown.cpp)
