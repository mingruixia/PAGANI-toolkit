find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME pytest_suite
           COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set(_env "PAGANI_BIN=${PAGANI_PACKAGE_DIR};PAGANI_TESTBIN=${PAGANI_TESTBIN_DIR}")
  if(TARGET test_gui_pipeline)
    list(APPEND _env "PAGANI_GUI_TEST=$<TARGET_FILE:test_gui_pipeline>")
  endif()
  set_tests_properties(pytest_suite PROPERTIES ENVIRONMENT "${_env}")
endif()
