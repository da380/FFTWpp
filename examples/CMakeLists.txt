# Each example verifies its own round trips and reports the result through
# its exit status, so they are registered as tests too: an example that is
# only ever compiled stops being a working demonstration soon enough.
foreach(_example Example1 Example2 Example3 Example4)
  add_executable(${_example} ${_example}.cpp)
  target_link_libraries(${_example} PRIVATE FFTWpp::FFTWpp
                                          FFTWppWarnings)
  if(FFTWPP_BUILD_TESTS)
    add_test(NAME ${_example} COMMAND ${_example})
  endif()
endforeach()
