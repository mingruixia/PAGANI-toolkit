add_library(pagani_common STATIC common.cpp nifti.cpp graph.cpp random_network.cpp)
target_include_directories(pagani_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pagani_common PUBLIC Threads::Threads)
