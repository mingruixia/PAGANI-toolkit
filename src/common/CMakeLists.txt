# zlib, for .nii.gz images: built from source as a static library so that the
# portable packages need no system library.  Only the sources are used
# (SOURCE_SUBDIR points nowhere): zlib's own CMakeLists does not support CMake 4.
include(FetchContent)
FetchContent_Declare(zlib
  URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
  URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
  SOURCE_SUBDIR no-cmake)
FetchContent_MakeAvailable(zlib)
enable_language(C)
set(_z ${zlib_SOURCE_DIR})
add_library(pagani_zlib STATIC
  ${_z}/adler32.c ${_z}/compress.c ${_z}/crc32.c ${_z}/deflate.c ${_z}/gzclose.c ${_z}/gzlib.c
  ${_z}/gzread.c ${_z}/gzwrite.c ${_z}/infback.c ${_z}/inffast.c ${_z}/inflate.c ${_z}/inftrees.c
  ${_z}/trees.c ${_z}/uncompr.c ${_z}/zutil.c)
target_include_directories(pagani_zlib SYSTEM PUBLIC ${_z})
if(MSVC)
  target_compile_definitions(pagani_zlib PRIVATE _CRT_SECURE_NO_DEPRECATE _CRT_NONSTDC_NO_DEPRECATE)
else()
  target_compile_definitions(pagani_zlib PRIVATE HAVE_UNISTD_H HAVE_STDARG_H)
endif()
set_target_properties(pagani_zlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pagani_common STATIC common.cpp nifti.cpp graph.cpp random_network.cpp)
target_include_directories(pagani_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pagani_common PUBLIC Threads::Threads PRIVATE pagani_zlib)
target_compile_definitions(pagani_common PRIVATE PAGANI_TOOLKIT_VERSION="${PROJECT_VERSION}")
