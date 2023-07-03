add_library(FFTWpp INTERFACE
  FFTWConcepts.h
  FFTWMemory.h
  FFTWFlags.h
  FFTWPlan.h
  FFTW.h
)

target_include_directories(FFTWpp INTERFACE ${FFTW_INCLUDES})
target_link_libraries(FFTWpp INTERFACE ${FFTW_LIBRARIES})


target_include_directories(FFTWpp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/>
  )
