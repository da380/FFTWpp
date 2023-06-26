add_library(FFTW++ INTERFACE
  FFTWAllocator.h
)

target_include_directories(FFTW++ INTERFACE ${FFTW_INCLUDES})
target_link_libraries(FFTW++ INTERFACE ${FFTW_LIBRARIES})


target_include_directories(FFTW++ INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/FFTW++>
  )
