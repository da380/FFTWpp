cmake_minimum_required(VERSION 3.19)

project(
  FFTWpp
  VERSION 1.0.0
  DESCRIPTION "A modern C++ interface to the FFTW3 library"
  HOMEPAGE_URL "https://github.com/da380/FFTWpp"
  LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds are not allowed. Make a separate build directory and "
      "run CMake from there. You may need to remove CMakeCache.txt first.")
endif()

# True when FFTWpp is the top-level project rather than a dependency of one.
# Only then do we build examples and tests, or install anything by default.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(FFTWPP_IS_TOP_LEVEL ON)
else()
  set(FFTWPP_IS_TOP_LEVEL OFF)
endif()

if(FFTWPP_IS_TOP_LEVEL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CMakeDependentOption)
include(GNUInstallDirs)

option(FFTWPP_BUILD_EXAMPLES "Build the FFTWpp examples"
       ${FFTWPP_IS_TOP_LEVEL})
option(FFTWPP_BUILD_TESTS "Build the FFTWpp tests" ${FFTWPP_IS_TOP_LEVEL})
option(FFTWPP_BUILD_DOCS "Build the Doxygen documentation" OFF)
option(FFTWPP_INSTALL "Generate FFTWpp's install and export rules"
       ${FFTWPP_IS_TOP_LEVEL})
option(FFTWPP_WARNINGS_AS_ERRORS
       "Treat compiler warnings in FFTWpp's own targets as errors" OFF)
option(FFTWPP_USE_FFTW_THREADS
       "Link FFTW's threading libraries and enable FFTWpp's threading wrappers"
       OFF)
set(FFTWPP_THREAD_BACKEND
    "Threads"
    CACHE STRING
          "Which FFTW threading libraries to link: Threads (pthreads) or OpenMP")
set_property(CACHE FFTWPP_THREAD_BACKEND PROPERTY STRINGS Threads OpenMP)

# The option names below were used before this project had a namespaced set.
# Honour them so that an existing build tree or CI script keeps working.
foreach(_legacy EXAMPLES TESTS)
  if(DEFINED MY_PROJECT_BUILD_${_legacy})
    message(
      DEPRECATION
        "MY_PROJECT_BUILD_${_legacy} is deprecated; use "
        "FFTWPP_BUILD_${_legacy} instead.")
    set(FFTWPP_BUILD_${_legacy} ${MY_PROJECT_BUILD_${_legacy}})
  endif()
endforeach()
if(DEFINED BUILD_DOCS)
  set(FFTWPP_BUILD_DOCS ${BUILD_DOCS})
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# FFTWpp dispatches on precision at compile time, so all three serial
# libraries are required whichever precision a given consumer happens to use.
set(FFTWPP_FFTW_COMPONENTS Double Float LongDouble)
if(FFTWPP_USE_FFTW_THREADS)
  foreach(_precision Double Float LongDouble)
    list(APPEND FFTWPP_FFTW_COMPONENTS ${_precision}${FFTWPP_THREAD_BACKEND})
  endforeach()
endif()
find_package(FFTW REQUIRED COMPONENTS ${FFTWPP_FFTW_COMPONENTS})

# FIND_PACKAGE_ARGS makes FetchContent try find_package first, so an installed
# copy is used when one exists and only an absent dependency is downloaded.
# Without it every consumer re-downloads and rebuilds, and this project cannot
# be packaged.
include(FetchContent)
FetchContent_Declare(
  NumericConcepts
  GIT_REPOSITORY https://github.com/da380/NumericConcepts.git
  GIT_TAG main
  FIND_PACKAGE_ARGS NAMES NumericConcepts)
FetchContent_MakeAvailable(NumericConcepts)
if(NOT TARGET NumericConcepts::NumericConcepts)
  add_library(NumericConcepts::NumericConcepts ALIAS NumericConcepts)
endif()

# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------

add_library(FFTWpp INTERFACE)
add_library(FFTWpp::FFTWpp ALIAS FFTWpp)

target_compile_features(FFTWpp INTERFACE cxx_std_20)

target_link_libraries(FFTWpp INTERFACE NumericConcepts::NumericConcepts)
foreach(_component IN LISTS FFTWPP_FFTW_COMPONENTS)
  target_link_libraries(FFTWpp INTERFACE FFTW::${_component})
endforeach()

if(FFTWPP_USE_FFTW_THREADS)
  target_compile_definitions(FFTWpp INTERFACE FFTWPP_ENABLE_THREADS)
endif()

target_include_directories(
  FFTWpp INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# ---------------------------------------------------------------------------
# Install and export
# ---------------------------------------------------------------------------

if(FFTWPP_INSTALL)
  include(CMakePackageConfigHelpers)
  set(FFTWPP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/FFTWpp")

  install(TARGETS FFTWpp EXPORT FFTWppTargets)

  # The public headers are included as <FFTWpp/Ranges>, so the whole FFTWpp
  # directory goes to the include prefix as it stands.
  install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/FFTWpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "Core"
    PATTERN "Ranges")

  install(
    EXPORT FFTWppTargets
    FILE FFTWppTargets.cmake
    NAMESPACE FFTWpp::
    DESTINATION "${FFTWPP_CMAKE_INSTALL_DIR}")

  # FFTW has no config file of its own on most systems, so the consumer needs
  # our find module to resolve the FFTW::* targets the export refers to.
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindFFTW.cmake"
          DESTINATION "${FFTWPP_CMAKE_INSTALL_DIR}")

  set(FFTWPP_EXPORTED_FFTW_COMPONENTS "${FFTWPP_FFTW_COMPONENTS}")
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FFTWppConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/FFTWppConfig.cmake"
    INSTALL_DESTINATION "${FFTWPP_CMAKE_INSTALL_DIR}")

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FFTWppConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/FFTWppConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/FFTWppConfigVersion.cmake"
          DESTINATION "${FFTWPP_CMAKE_INSTALL_DIR}")
endif()

# ---------------------------------------------------------------------------
# Examples, tests and documentation
# ---------------------------------------------------------------------------

# Warning flags for FFTWpp's own executables. Deliberately not part of the
# FFTWpp interface target: a consumer chooses its own warning policy, and a
# header-only library has no business imposing one.
add_library(FFTWppWarnings INTERFACE)
if(MSVC)
  target_compile_options(FFTWppWarnings INTERFACE /W4
                         $<$<BOOL:${FFTWPP_WARNINGS_AS_ERRORS}>:/WX>)
else()
  target_compile_options(FFTWppWarnings INTERFACE -Wall -Wextra
                         $<$<BOOL:${FFTWPP_WARNINGS_AS_ERRORS}>:-Werror>)
endif()

if(FFTWPP_BUILD_TESTS)
  enable_testing()
endif()

if(FFTWPP_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(FFTWPP_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(FFTWPP_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
  configure_file(Doxyfile.in "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile" @ONLY)
  add_custom_target(
    doc ALL
    COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
endif()
