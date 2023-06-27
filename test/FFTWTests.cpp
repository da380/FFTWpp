#include <gtest/gtest.h>

#include "FFTW1DTest.h"
#include "FFTW1DRealTest.h"

// Tests for 1D complex transformations

TEST(FFTW1DTest, Float) {
  EXPECT_EQ(0, FFTW1DTest<float>());
  EXPECT_EQ(0, FFTW1DTest<float>(true));
}

TEST(FFTW1DTest, Double) {
  EXPECT_EQ(0, FFTW1DTest<double>());
  EXPECT_EQ(0, FFTW1DTest<double>(true));
}

TEST(FFTW1DTest, LongDouble) {
  EXPECT_EQ(0, FFTW1DTest<long double>());
  EXPECT_EQ(0, FFTW1DTest<long double>(true));
}


// Tests for 1D real-to-complex transformations.

TEST(FFTW1DRealTest, Float) {
  EXPECT_EQ(0, FFTW1DRealTest<float>());
  EXPECT_EQ(0, FFTW1DRealTest<float>(true));
}

TEST(FFTW1DRealTest, Double) {
  EXPECT_EQ(0, FFTW1DRealTest<double>());
  EXPECT_EQ(0, FFTW1DRealTest<double>(true));
}


TEST(FFTW1DRealTest, LongDouble) {
  EXPECT_EQ(0, FFTW1DRealTest<long double>());
  EXPECT_EQ(0, FFTW1DRealTest<long double>(true));
}

