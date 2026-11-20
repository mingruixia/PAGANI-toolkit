add_library(pagani_metrics STATIC metrics.cpp local_efficiency.cpp weighted_betweenness.cpp)
target_include_directories(pagani_metrics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pagani_metrics PUBLIC pagani_common)
