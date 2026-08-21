# Each example verifies its own behaviour and reports the result through its
# exit status, so they are registered as tests too: an example that is only
# ever compiled stops being a working demonstration soon enough.
#
# The numbering is the order they are meant to be read in.
set(FFTWPP_EXAMPLES
    01-getting_started
    02-transform_types
    03-many_dimensions
    04-batched_transforms
    05-wisdom
    06-guru_layouts
    07-plan_reuse
    08-threads)

foreach(_example IN LISTS FFTWPP_EXAMPLES)
  add_executable(${_example} ${_example}.cpp)
  target_link_libraries(${_example} PRIVATE FFTWpp::FFTWpp FFTWppWarnings)
  if(FFTWPP_BUILD_TESTS)
    add_test(NAME ${_example} COMMAND ${_example})
  endif()
endforeach()
