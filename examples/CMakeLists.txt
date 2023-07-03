add_executable(FFTWConceptsTest
               FFTWConceptsTest.cpp)
	     target_link_libraries(FFTWConceptsTest PRIVATE FFTWpp)


add_executable(fftw
               fftw.cpp)
	     target_link_libraries(fftw PRIVATE FFTWpp)	     




