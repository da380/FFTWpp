
cmake_minimum_required (VERSION 3.5)


project(FFTWpp
        VERSION 1.0
        LANGUAGES CXX
        )

enable_testing()
      
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Locate and set up FFTW3
find_package(FFTW)

add_subdirectory(include)
add_subdirectory(examples)
add_subdirectory(test)








