# A standalone project that consumes an installed FFTWpp through
# find_package. It exists so that CI notices if the install or export rules
# stop producing a usable package; it is deliberately not part of the main
# build.
cmake_minimum_required(VERSION 3.19)
project(FFTWppPackageTest LANGUAGES CXX)

find_package(FFTWpp REQUIRED)

add_executable(PackageTest main.cpp)
target_link_libraries(PackageTest PRIVATE FFTWpp::FFTWpp)

enable_testing()
add_test(NAME PackageTest COMMAND PackageTest)
