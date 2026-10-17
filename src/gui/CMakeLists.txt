find_package(QT NAMES Qt6 Qt5 COMPONENTS Core Widgets)
if(NOT QT_FOUND)
  message(WARNING "Qt not found: the GUI will not be built (set CMAKE_PREFIX_PATH to your Qt installation)")
  return()
endif()
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Widgets)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

add_library(pagani_gui_core STATIC pipeline.cpp pipeline.hpp)
target_link_libraries(pagani_gui_core PUBLIC Qt${QT_VERSION_MAJOR}::Core)
target_include_directories(pagani_gui_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(GUI_SOURCES main.cpp mainwindow.cpp mainwindow.hpp resources.qrc)
if(WIN32)
  list(APPEND GUI_SOURCES pagani.rc)
endif()
add_executable(pagani_gui WIN32 ${GUI_SOURCES})
target_link_libraries(pagani_gui PRIVATE pagani_gui_core Qt${QT_VERSION_MAJOR}::Widgets)
target_compile_definitions(pagani_gui PRIVATE PAGANI_VERSION="${PROJECT_VERSION}")
set_target_properties(pagani_gui PROPERTIES OUTPUT_NAME PAGANI
  RUNTIME_OUTPUT_DIRECTORY ${PAGANI_PACKAGE_DIR}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PAGANI_PACKAGE_DIR}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PAGANI_PACKAGE_DIR})
install(TARGETS pagani_gui RUNTIME DESTINATION .)

if(PAGANI_BUILD_TESTS)
  add_executable(test_gui_pipeline test_pipeline.cpp)
  target_link_libraries(test_gui_pipeline PRIVATE pagani_gui_core)
  add_test(NAME gui_pipeline COMMAND test_gui_pipeline ${CMAKE_SOURCE_DIR}/legacy/software_package-2.0)
endif()
