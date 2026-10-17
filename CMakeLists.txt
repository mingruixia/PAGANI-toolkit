cmake_minimum_required(VERSION 3.21)
project(PAGANI_Toolkit VERSION 2.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PAGANI_BUILD_CUDA  "Build the GPU tools (requires the CUDA toolkit)" ON)
option(PAGANI_BUILD_GUI   "Build the Qt GUI" ON)
option(PAGANI_BUILD_TESTS "Build unit tests" ON)

if(MSVC)
  # Static runtime: the portable package must not depend on the VC++ redistributable.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

find_package(Threads REQUIRED)

# Layout of the portable package produced by the build:
#   <build>/package/PAGANI.exe, exefiles/, exefiles_weighted/
set(PAGANI_PACKAGE_DIR ${CMAKE_BINARY_DIR}/package)
set(PAGANI_UNWEIGHTED_DIR ${PAGANI_PACKAGE_DIR}/exefiles)
set(PAGANI_WEIGHTED_DIR ${PAGANI_PACKAGE_DIR}/exefiles_weighted)
# CPU-backend builds of the GPU tools, used only by the test-suite
set(PAGANI_TESTBIN_DIR ${CMAKE_BINARY_DIR}/test-bin)

if(PAGANI_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(src/common)
add_subdirectory(src/metrics)
add_subdirectory(src/construct)
add_subdirectory(src/apps)

if(PAGANI_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    add_subdirectory(src/gpu)
  else()
    message(WARNING "CUDA compiler not found: GPU tools will not be built")
  endif()
endif()

if(PAGANI_BUILD_GUI)
  add_subdirectory(src/gui)
endif()

if(PAGANI_BUILD_TESTS)
  add_subdirectory(tests)
endif()
