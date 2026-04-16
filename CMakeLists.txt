# CMake build for MinAI. Works on macOS, Linux, and Windows (MSVC or clang-cl).
#
# Typical usage:
#   cmake -B build
#   cmake --build build --config Release
#
# The executable lands at build/minai (or build/Release/minai.exe on MSVC).

cmake_minimum_required(VERSION 3.14)
project(minai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(minai minai.cpp)

if(MSVC)
    # /utf-8 so embedded Unicode (UTF-8 block chars in the sparkline hex
    # escapes, em-dashes in comments) is interpreted correctly.
    target_compile_options(minai PRIVATE /W4 /EHsc /utf-8)
else()
    target_compile_options(minai PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()
