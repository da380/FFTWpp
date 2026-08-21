foreach(_example Example1 Example2 Example3 Example4)
  add_executable(${_example} ${_example}.cpp)
  target_link_libraries(${_example} PRIVATE FFTWpp::FFTWpp)
endforeach()
