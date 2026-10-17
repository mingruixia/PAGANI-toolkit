find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME cpu_tools
           COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/test_cpu_tools.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(cpu_tools PROPERTIES ENVIRONMENT "PAGANI_BIN=${PAGANI_PACKAGE_DIR}")
endif()
