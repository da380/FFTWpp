include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  FIND_PACKAGE_ARGS NAMES GTest)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)

if(NOT TARGET GTest::gtest)
  add_library(GTest::gtest ALIAS gtest)
endif()

add_executable(Tests OdrInclude.cpp TestMain.cpp Tests.cpp)
target_link_libraries(Tests PRIVATE FFTWpp::FFTWpp GTest::gtest)

include(GoogleTest)
gtest_discover_tests(Tests)
