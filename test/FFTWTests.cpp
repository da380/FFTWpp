#include <gtest/gtest.h>
#include "FFTW1DTest.h"



// Tests for 1D complex transformations

TEST(FFTWTest1D,Float) {
  EXPECT_EQ(0, FFTW1DTestHelper<float>());
}

TEST(FFTWTest1D,Double) {
  EXPECT_EQ(0, FFTW1DTestHelper<double>());
}

TEST(FFTWTest1D,Quadruple) {
  EXPECT_EQ(0, FFTW1DTestHelper<long double>());
}



