include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.11.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)



add_executable(FFTWTests
               FFTWTests.cpp)
target_link_libraries(FFTWTests PRIVATE FFTW  gtest_main)
include(GoogleTest)
gtest_discover_tests(FFTWTests)
